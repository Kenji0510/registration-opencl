use anyhow::{Context, Result};
use ocl::{Buffer, Kernel, MemFlags, OclPrm, Program, Queue};

use crate::{gpu_voxel::OclVoxelContext, ocl_context::OclRuntime};

const KERNEL_SRC: &str = include_str!("kernels/search.cl");

fn round_up(x: usize, multiple: usize) -> usize {
    if x % multiple == 0 {
        x
    } else {
        (x / multiple + 1) * multiple
    }
}

pub struct OclSearchContext {
    rt: OclRuntime,
    program: Program,
    kernel_func: Kernel,

    buf_indices: Option<Buffer<i32>>,
    buf_dists: Option<Buffer<f32>>,
}

impl OclSearchContext {
    pub fn new(rt: OclRuntime) -> Result<Self> {
        let program = Program::builder()
            .src(KERNEL_SRC)
            .devices(rt.device.clone())
            .build(&rt.context)
            .context("Program build failed")?;

        let dummy_u64 = Buffer::<u64>::builder()
            .queue(rt.queue.clone())
            .flags(MemFlags::new().read_write())
            .len(1)
            .build()
            .context("Failed to create dummy_u64")?;

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
            .name("find_nearest_neighbor")
            .queue(rt.queue.clone())
            .global_work_size(1)
            .local_work_size(1)
            .arg(&dummy_f32) // Source points
            .arg(0) // num_source_points
            .arg(&dummy_f32) // Target points
            .arg(0) // num_target_points
            .arg(&dummy_i32) // out_indices
            .arg(&dummy_f32) // out_dists
            .build()
            .context("Failed to set the kernel")?;

        Ok(Self {
            rt,
            program,
            kernel_func,
            buf_indices: None,
            buf_dists: None,
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

    pub fn compute_find_nearest_neighbor(
        &mut self,
        source_pts: &Buffer<f32>,
        num_source: usize,
        target_pts: &Buffer<f32>,
        num_target: usize,
    ) -> Result<(Buffer<i32>, Buffer<f32>, Vec<i32>, Vec<f32>)> {
        if num_source == 0 {
            anyhow::bail!("Number of source points is zero");
        }

        let q = self.rt.queue.clone();
        Self::ensure_buffer(
            &q,
            &mut self.buf_indices,
            num_source,
            MemFlags::new().read_write(),
        )?;
        Self::ensure_buffer(
            &q,
            &mut self.buf_dists,
            num_source,
            MemFlags::new().read_write(),
        )?;

        let d_indices = self.buf_indices.as_mut().unwrap();
        let d_dists = self.buf_dists.as_mut().unwrap();

        let lws = 256 as usize;
        let gws = round_up(num_source, lws);

        self.kernel_func
            .set_arg(0, source_pts)
            .context("Failed to set source_pts arg")?;
        self.kernel_func
            .set_arg(1, num_source as u32)
            .context("Failed to set num_source_points arg")?;
        self.kernel_func
            .set_arg(2, target_pts)
            .context("Failed to set target_pts arg")?;
        self.kernel_func
            .set_arg(3, num_target as u32)
            .context("Failed to set num_target_points arg")?;
        self.kernel_func
            .set_arg(4, d_indices)
            .context("Failed to set out_indices arg")?;
        self.kernel_func
            .set_arg(5, d_dists)
            .context("Failed to set out_dists arg")?;

        unsafe {
            self.kernel_func
                .cmd()
                .global_work_size(gws)
                .local_work_size(lws)
                .enq()
                .context("Failed to enqueue kernel")?;
        }

        q.finish().context("Failed to finish queue")?;

        let d_indices = self.buf_indices.as_ref().unwrap().clone();
        let d_dists = self.buf_dists.as_ref().unwrap().clone();

        let mut indices = vec![0i32; num_source];
        let mut dists = vec![0.0f32; num_source];
        d_indices
            .read(&mut indices)
            .enq()
            .context("Failed to read indices")?;
        d_dists
            .read(&mut dists)
            .enq()
            .context("Failed to read distances")?;
        q.finish().context("Failed to finish queue")?;

        Ok((d_indices, d_dists, indices, dists))
    }
}
