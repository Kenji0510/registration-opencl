use anyhow::{Context, Result};
use ndarray::Array2;
use ocl::{Buffer, Kernel, MemFlags, OclPrm, Program, Queue};

use crate::ocl_context::OclRuntime;

const KERNEL_SRC: &str = include_str!("kernels/transform.cl");

fn round_up(x: usize, multiple: usize) -> usize {
    if x % multiple == 0 {
        x
    } else {
        (x / multiple + 1) * multiple
    }
}

pub struct OclTransformContext {
    rt: OclRuntime,
    program: Program,
    kernel_func: Kernel,

    buf_out_pts: Option<Buffer<f32>>,
    buf_out_covs: Option<Buffer<f32>>,
}

impl OclTransformContext {
    pub fn new(rt: OclRuntime) -> Result<Self> {
        let program = Program::builder()
            .src(KERNEL_SRC)
            .devices(rt.device.clone())
            .build(&rt.context)
            .context("Program build failed")?;

        let dummy_f32 = Buffer::<f32>::builder()
            .queue(rt.queue.clone())
            .flags(MemFlags::new().read_write())
            .len(1)
            .build()
            .context("Failed to create dummy_f32")?;

        let kernel_func = Kernel::builder()
            .program(&program)
            .name("transform_points_and_covs")
            .queue(rt.queue.clone())
            .global_work_size(1)
            .local_work_size(1)
            .arg(&dummy_f32) // pts
            .arg(&dummy_f32) // covs
            .arg(0) // num_points
            .arg(0.0f32) // r00
            .arg(0.0f32) // r01
            .arg(0.0f32) // r02
            .arg(0.0f32) // t0
            .arg(0.0f32) // r10
            .arg(0.0f32) // r11
            .arg(0.0f32) // r12
            .arg(0.0f32) // t1
            .arg(0.0f32) // r20
            .arg(0.0f32) // r21
            .arg(0.0f32) // r22
            .arg(0.0f32) // t2
            .arg(&dummy_f32) // out_pts
            .arg(&dummy_f32) // out_covs
            .build()
            .context("Failed to set the kernel")?;

        Ok(Self {
            rt,
            program,
            kernel_func,
            buf_out_pts: None,
            buf_out_covs: None,
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

    pub fn apply_transform(
        &mut self,
        pts: &Buffer<f32>,
        covs: &Buffer<f32>,
        num_points: usize,
        transform: &Array2<f32>,
    ) -> Result<(Buffer<f32>, Buffer<f32>)> {
        let q = self.rt.queue.clone();
        Self::ensure_buffer(
            &q,
            &mut self.buf_out_pts,
            num_points * 3,
            MemFlags::new().read_write(),
        )?;
        Self::ensure_buffer(
            &q,
            &mut self.buf_out_covs,
            num_points * 9,
            MemFlags::new().read_write(),
        )?;

        let d_out_pts = self.buf_out_pts.as_ref().unwrap();
        let d_out_covs = self.buf_out_covs.as_ref().unwrap();

        let lws = 256 as usize;
        let gws = round_up(num_points, lws);

        self.kernel_func
            .set_arg(0, pts)
            .context("Failed to set pts arg")?;
        self.kernel_func
            .set_arg(1, covs)
            .context("Failed to set covs arg")?;
        self.kernel_func
            .set_arg(2, num_points as i32)
            .context("Failed to set num_points arg")?;
        self.kernel_func
            .set_arg(3, transform[[0, 0]])
            .context("Failed to set r00 arg")?;
        self.kernel_func
            .set_arg(4, transform[[0, 1]])
            .context("Failed to set r01 arg")?;
        self.kernel_func
            .set_arg(5, transform[[0, 2]])
            .context("Failed to set r02 arg")?;
        self.kernel_func
            .set_arg(6, transform[[0, 3]])
            .context("Failed to set t0 arg")?;
        self.kernel_func
            .set_arg(7, transform[[1, 0]])
            .context("Failed to set r10 arg")?;
        self.kernel_func
            .set_arg(8, transform[[1, 1]])
            .context("Failed to set r11 arg")?;
        self.kernel_func
            .set_arg(9, transform[[1, 2]])
            .context("Failed to set r12 arg")?;
        self.kernel_func
            .set_arg(10, transform[[1, 3]])
            .context("Failed to set t1 arg")?;
        self.kernel_func
            .set_arg(11, transform[[2, 0]])
            .context("Failed to set r20 arg")?;
        self.kernel_func
            .set_arg(12, transform[[2, 1]])
            .context("Failed to set r21 arg")?;
        self.kernel_func
            .set_arg(13, transform[[2, 2]])
            .context("Failed to set r22 arg")?;
        self.kernel_func
            .set_arg(14, transform[[2, 3]])
            .context("Failed to set t2 arg")?;
        self.kernel_func
            .set_arg(15, d_out_pts)
            .context("Failed to set out_pts arg")?;
        self.kernel_func
            .set_arg(16, d_out_covs)
            .context("Failed to set out_covs arg")?;

        unsafe {
            self.kernel_func
                .cmd()
                .global_work_size(gws)
                .local_work_size(lws)
                .enq()
                .context("Failed to enqueue kernel")?;
        }

        q.finish().context("Failed to finish queue")?;

        // Create new buffers and copy the results to avoid overwriting on next call
        let result_pts = Buffer::<f32>::builder()
            .queue(self.rt.queue.clone())
            .flags(MemFlags::new().read_write())
            .len(num_points * 3)
            .build()
            .context("Failed to create result pts buffer")?;

        let result_covs = Buffer::<f32>::builder()
            .queue(self.rt.queue.clone())
            .flags(MemFlags::new().read_write())
            .len(num_points * 9)
            .build()
            .context("Failed to create result covs buffer")?;

        d_out_pts.cmd()
            .copy(&result_pts, Some(0), Some(num_points * 3))
            .enq()
            .context("Failed to copy transformed pts buffer")?;

        d_out_covs.cmd()
            .copy(&result_covs, Some(0), Some(num_points * 9))
            .enq()
            .context("Failed to copy transformed covs buffer")?;

        q.finish()?;

        Ok((result_pts, result_covs))
    }
}
