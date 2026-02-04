use anyhow::{Context, Result};
use ocl::{Buffer, Kernel, MemFlags, OclPrm, ProQue, Program, Queue};

use crate::ocl_context::OclRuntime;

const KERNEL_SRC: &str = include_str!("kernels/compute_normals.cl");

fn round_up(x: usize, multiple: usize) -> usize {
    if x % multiple == 0 {
        x
    } else {
        (x / multiple + 1) * multiple
    }
}

pub struct OclNormalsContext {
    rt: OclRuntime,
    program: Program,
    kernel_func: Kernel,

    buf_points: Option<Buffer<f32>>,
    buf_normals: Option<Buffer<f32>>,
}

impl OclNormalsContext {
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
            .name("compute_normals")
            .queue(rt.queue.clone())
            .global_work_size(1)
            .local_work_size(1)
            .arg(&dummy_f32) // points
            .arg(&dummy_i32) // neighbor_indices
            .arg(0) // num_points
            .arg(0) // vp_x (Viewpoint x)
            .arg(0) // vp_y (Viewpoint y)
            .arg(0) // vp_z (Viewpoint z)
            .arg(&dummy_f32) // out_normals
            .build()?;

        Ok(Self {
            rt,
            program,
            kernel_func,
            buf_points: None,
            buf_normals: None,
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

    pub fn compute_normals(
        &mut self,
        points_buffer: &Buffer<f32>,
        num_points: usize,
        neighbors_indices_buffer: &Buffer<i32>,
        vp_x: f32,
        vp_y: f32,
        vp_z: f32,
    ) -> Result<Buffer<f32>> {
        if num_points == 0 {
            anyhow::bail!("Number of points is zero");
        }

        let q = self.rt.queue.clone();

        Self::ensure_buffer(
            &q,
            &mut self.buf_normals,
            num_points * 3,
            MemFlags::new().read_write(),
        )
        .context("Failed to ensure normals buffer")?;
        let d_normals = self.buf_normals.as_ref().unwrap();

        let lws = 256 as usize;
        let gws = round_up(num_points, lws);

        self.kernel_func
            .set_arg(0, points_buffer)
            .context("Failed to set points_buffer arg")?;
        self.kernel_func
            .set_arg(1, neighbors_indices_buffer)
            .context("Failed to set neighbors_indices_buffer arg")?;
        self.kernel_func
            .set_arg(2, num_points as i32)
            .context("Failed to set num_points arg")?;
        self.kernel_func
            .set_arg(3, vp_x)
            .context("Failed to set vp_x arg")?;
        self.kernel_func
            .set_arg(4, vp_y)
            .context("Failed to set vp_y arg")?;
        self.kernel_func
            .set_arg(5, vp_z)
            .context("Failed to set vp_z arg")?;
        self.kernel_func
            .set_arg(6, d_normals)
            .context("Failed to set out_normals arg")?;

        unsafe {
            self.kernel_func
                .cmd()
                .global_work_size(gws)
                .local_work_size(lws)
                .enq()
                .context("Failed to enqueue kernel")?;
        }

        q.finish().context("Failed to finish queue")?;

        Ok(d_normals.clone())
    }
}
