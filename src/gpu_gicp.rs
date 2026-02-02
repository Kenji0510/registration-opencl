use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ocl::{Buffer, Kernel, MemFlags, OclPrm, Program, Queue};

use crate::ocl_context::OclRuntime;

const KERNEL_SRC: &str = include_str!("kernels/gicp.cl");

fn round_up(x: usize, multiple: usize) -> usize {
    if x % multiple == 0 {
        x
    } else {
        (x / multiple + 1) * multiple
    }
}

pub struct OclGicpContext {
    rt: OclRuntime,
    program: Program,
    kernel_func: Kernel,

    buf_h: Option<Buffer<f32>>,
    buf_b: Option<Buffer<f32>>,
}

impl OclGicpContext {
    pub fn new(rt: OclRuntime) -> Result<Self> {
        let program = Program::builder()
            .src(KERNEL_SRC)
            .devices(rt.device.clone())
            .cmplr_opt("-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros")
            .build(&rt.context)
            .context("Program build failed")?;

        let dummy_f32 = Buffer::<f32>::builder()
            .queue(rt.queue.clone())
            .flags(MemFlags::new().read_write())
            .len(1)
            .build()
            .context("Failed to create dummy_f32")?;

        let dummy_i32 = Buffer::<i32>::builder()
            .queue(rt.queue.clone())
            .flags(MemFlags::new().read_write())
            .len(1)
            .build()
            .context("Failed to create dummy_i32")?;

        let kernel_func = Kernel::builder()
            .program(&program)
            .name("compute_gicp_linear_system")
            .queue(rt.queue.clone())
            .global_work_size(1)
            .local_work_size(1)
            .arg(&dummy_f32) // num_source * 3
            .arg(&dummy_f32) // num_source * 9
            .arg(&dummy_f32) // num_target * 3
            .arg(&dummy_f32) // num_target * 9
            .arg(&dummy_i32) // source_indices
            .arg(&dummy_f32) // source_dists_sq
            .arg(0) // num_source
            .arg(0) // num_target
            .arg(0.0f32) // max_dist_sq
            .arg(&dummy_f32) // d_H
            .arg(&dummy_f32) // d_b
            .build()?;

        Ok(Self {
            rt,
            program,
            kernel_func,
            buf_h: None,
            buf_b: None,
        })
    }

    fn ensure_buffer<T: OclPrm>(
        queue: &Queue,
        buf: &mut Option<Buffer<T>>,
        len_needed: usize,
        flags: MemFlags,
    ) -> Result<()> {
        let cur = buf.as_ref().map(|b| b.len()).unwrap_or(0);
        if cur < len_needed {
            let new_len = ((len_needed as f32) * 1.2).ceil() as usize;
            *buf = Some(
                Buffer::<T>::builder()
                    .queue(queue.clone())
                    .flags(flags)
                    .len(new_len)
                    .build()
                    .context("Failed to create buffer")?,
            );
        }
        Ok(())
    }

    pub fn compute_gicp(
        &mut self,
        d_source_pts: &Buffer<f32>,
        d_source_covs: &Buffer<f32>,
        num_source: usize,
        d_target_pts: &Buffer<f32>,
        d_target_covs: &Buffer<f32>,
        num_target: usize,
        d_indices: &Buffer<i32>,
        d_distances: &Buffer<f32>,
        max_dist_sq: f32,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        println!("GICP: num_source = {}, num_target = {}", num_source, num_target);

        if num_source == 0 || num_target == 0 {
            anyhow::bail!("Empty point cloud");
        }

        let queue = self.rt.queue.clone();

        Self::ensure_buffer(&queue, &mut self.buf_h, 36, MemFlags::new().read_write())?;

        Self::ensure_buffer(&queue, &mut self.buf_b, 6, MemFlags::new().read_write())?;

        {
            let d_h = self.buf_h.as_ref().unwrap();
            let d_b = self.buf_b.as_ref().unwrap();

            let lws = 64 as usize;
            let gws = round_up(num_source, lws);

            self.kernel_func
                .set_arg(0, d_source_pts)
                .context("Failed to set d_source_pts arg")?;
            self.kernel_func
                .set_arg(1, d_source_covs)
                .context("Failed to set d_source_covs arg")?;
            self.kernel_func
                .set_arg(2, d_target_pts)
                .context("Failed to set d_target_pts arg")?;
            self.kernel_func
                .set_arg(3, d_target_covs)
                .context("Failed to set d_target_covs arg")?;
            self.kernel_func
                .set_arg(4, d_indices)
                .context("Failed to set d_indices arg")?;
            self.kernel_func
                .set_arg(5, d_distances)
                .context("Failed to set d_distances arg")?;
            self.kernel_func
                .set_arg(6, num_source as i32)
                .context("Failed to set num_source arg")?;
            self.kernel_func
                .set_arg(7, num_target as i32)
                .context("Failed to set num_target arg")?;
            self.kernel_func
                .set_arg(8, max_dist_sq)
                .context("Failed to set max_dist_sq arg")?;
            self.kernel_func
                .set_arg(9, d_h)
                .context("Failed to set d_H arg")?;
            self.kernel_func
                .set_arg(10, d_b)
                .context("Failed to set d_b arg")?;

            unsafe {
                self.kernel_func
                    .cmd()
                    .global_work_size(gws)
                    .local_work_size(lws)
                    .enq()
                    .context("Failed to enqueue GICP kernel")?;
            }

            queue.finish().context("Failed to enqueue kernel")?;
        }

        let d_H = self.buf_h.as_ref().unwrap();
        let d_b = self.buf_b.as_ref().unwrap();

        let mut h_H = vec![0.0f32; 36];
        let mut h_b = vec![0.0f32; 6];

        d_H.read(&mut h_H)
            .enq()
            .context("Failed to read H matrix")?;
        d_b.read(&mut h_b)
            .enq()
            .context("Failed to read b vector")?;

        queue.finish().context("Failed to finish queue")?;

        let h_vec_f64 = h_H.iter().map(|&x| x as f64).collect::<Vec<f64>>();
        let h_b_f64 = h_b.iter().map(|&x| x as f64).collect::<Vec<f64>>();

        let h_matrix = Array2::from_shape_vec((6, 6), h_vec_f64)
            .context("Failed to create H matrix ndarray")?;
        let b_vector = Array1::from_vec(h_b_f64)
            .into_dimensionality::<ndarray::Ix1>()
            .context("Failed to create b vector ndarray")?;

        Ok((h_matrix, b_vector))
    }
}
