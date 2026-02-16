use std::time::Instant;

use anyhow::{Context, Result};
use ndarray::Array2;
use ocl::{Buffer, Kernel, MemFlags, OclPrm, ProQue, Program, Queue, flags};

use crate::ocl_context::OclRuntime;

const KERNEL_SRC: &str = include_str!("kernels/voxel.cl");

fn round_up(x: usize, multiple: usize) -> usize {
    if x % multiple == 0 {
        x
    } else {
        (x / multiple + 1) * multiple
    }
}

pub struct OclVoxelContext {
    rt: OclRuntime,
    program: Program,
    kernel_init: Kernel,
    kernel_insert: Kernel,
    kernel_average: Kernel,
    kernel_compact: Kernel,

    pub buf_table_keys: Option<Buffer<u64>>,
    pub buf_table_centroids: Option<Buffer<f32>>,
    pub buf_table_counts: Option<Buffer<i32>>,
    pub buf_table_remap: Option<Buffer<i32>>,
    pub buf_input_points: Option<Buffer<f32>>,
    pub buf_out_points: Option<Buffer<f32>>,
    pub buf_valid_count: Option<Buffer<i32>>,

    pub table_size: usize,
    pub voxel_size: f32,

    lws: usize,
}

impl OclVoxelContext {
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

        let lws = 256;

        let kernel_init = Kernel::builder()
            .program(&program)
            .name("init_table")
            .queue(rt.queue.clone())
            .global_work_size(1)
            .local_work_size(1)
            .arg(&dummy_u64) // table_keys
            .arg(&dummy_i32) // table_remap
            .arg(0i32) // table_size
            .build()?;

        let kernel_insert = Kernel::builder()
            .program(&program)
            .name("insert_points")
            .queue(rt.queue.clone())
            .global_work_size(1)
            .local_work_size(1)
            .arg(&dummy_f32) // points
            .arg(0i32) // num_points
            .arg(0.0f32) // voxel_size
            .arg(&dummy_u64) // table_keys
            .arg(&dummy_f32) // table_centroids
            .arg(&dummy_i32) // table_counts
            .arg(0i32) // table_size
            .build()?;

        let kernel_average = Kernel::builder()
            .program(&program)
            .name("average_table")
            .queue(rt.queue.clone())
            .global_work_size(1)
            .local_work_size(1)
            .arg(&dummy_f32) // table_centroids
            .arg(&dummy_i32) // table_counts
            .arg(0i32) // table_size
            .build()?;

        let kernel_compact = Kernel::builder()
            .program(&program)
            .name("compact_voxels")
            .queue(rt.queue.clone())
            .global_work_size(1)
            .local_work_size(1)
            .arg(&dummy_u64) // table_keys
            .arg(&dummy_f32) // table_centroids
            .arg(&dummy_i32) // table_counts
            .arg(&dummy_i32) // table_remap
            .arg(0i32) // table_size
            .arg(&dummy_f32) // out_points
            .arg(&dummy_i32) // out_count
            .build()?;

        Ok(Self {
            rt,
            program,
            kernel_init,
            kernel_insert,
            kernel_average,
            kernel_compact,
            buf_table_keys: None,
            buf_table_centroids: None,
            buf_table_counts: None,
            buf_table_remap: None,
            buf_input_points: None,
            buf_out_points: None,
            buf_valid_count: None,
            table_size: 0,
            voxel_size: 0.0,
            lws,
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

    pub fn voxel_downsample(
        &mut self,
        input_points: &Array2<f32>,
        num_points: usize,
        voxel_size: f32,
    ) -> Result<(Buffer<f32>, usize)> {
        if num_points == 0 {
            anyhow::bail!("No input points");
        }

        let table_size = num_points * 2;

        let queue = self.rt.queue.clone();

        Self::ensure_buffer(
            &queue,
            &mut self.buf_table_keys,
            table_size,
            MemFlags::new().read_write(),
        )?;
        Self::ensure_buffer(
            &queue,
            &mut self.buf_table_centroids,
            table_size * 3,
            MemFlags::new().read_write(),
        )?;
        Self::ensure_buffer(
            &queue,
            &mut self.buf_table_counts,
            table_size,
            MemFlags::new().read_write(),
        )?;
        Self::ensure_buffer(
            &queue,
            &mut self.buf_table_remap,
            table_size,
            MemFlags::new().read_write(),
        )?;
        Self::ensure_buffer(
            &queue,
            &mut self.buf_input_points,
            num_points * 3,
            MemFlags::new().read_only(),
        )?;
        Self::ensure_buffer(
            &queue,
            &mut self.buf_out_points,
            num_points * 3,
            MemFlags::new().read_write(),
        )?;
        Self::ensure_buffer(
            &queue,
            &mut self.buf_valid_count,
            1,
            MemFlags::new().read_write(),
        )?;

        self.table_size = table_size;
        self.voxel_size = voxel_size;

        {
            let d_keys = self.buf_table_keys.as_ref().unwrap();
            let d_centroids = self.buf_table_centroids.as_ref().unwrap();
            let d_counts = self.buf_table_counts.as_ref().unwrap();
            let d_remap = self.buf_table_remap.as_ref().unwrap();
            let d_input = self.buf_input_points.as_ref().unwrap();
            let d_output = self.buf_out_points.as_ref().unwrap();
            let d_counter = self.buf_valid_count.as_ref().unwrap();

            // HtoD転送
            let t_htod = Instant::now();
            d_input
                .write(&input_points.as_slice().unwrap()[..num_points * 3])
                .enq()
                .context("Failed to write input points")?;
            self.rt.queue.finish()?;
            let htod_ms = t_htod.elapsed().as_secs_f64() * 1000.0;
            d_counter
                .write(&[0i32][..])
                .enq()
                .context("Failed to write valid count")?;
            d_centroids
                .cmd()
                .fill(0.0f32, None)
                .enq()
                .context("Failed to fill centroids")?;
            d_counts
                .cmd()
                .fill(0i32, None)
                .enq()
                .context("Failed to fill counts")?;

            let gws_table = round_up(table_size, self.lws);
            let gws_pts = round_up(num_points, self.lws);

            self.kernel_init
                .set_default_global_work_size(ocl::SpatialDims::One(gws_table));
            self.kernel_init
                .set_default_local_work_size(ocl::SpatialDims::One(self.lws));

            // init_table
            // let start = Instant::now();
            // let t_init = Instant::now();
            self.kernel_init.set_arg(0, d_keys)?;
            self.kernel_init.set_arg(1, d_remap)?;
            self.kernel_init.set_arg(2, table_size as i32)?;
            unsafe {
                self.kernel_init
                    .enq()
                    .context("Failed to run init_table kernel")?;
            }
            // self.rt.queue.finish()?;
            // let init_ms = t_init.elapsed().as_secs_f64() * 1000.0;

            // insert_points
            // let t_insert = Instant::now();
            self.kernel_insert
                .set_default_global_work_size(ocl::SpatialDims::One(gws_pts));
            self.kernel_insert
                .set_default_local_work_size(ocl::SpatialDims::One(self.lws));

            self.kernel_insert.set_arg(0, d_input)?;
            self.kernel_insert.set_arg(1, &(num_points as i32))?;
            self.kernel_insert.set_arg(2, &voxel_size)?;
            self.kernel_insert.set_arg(3, d_keys)?;
            self.kernel_insert.set_arg(4, d_centroids)?;
            self.kernel_insert.set_arg(5, d_counts)?;
            self.kernel_insert.set_arg(6, &(table_size as i32))?;
            unsafe {
                self.kernel_insert.enq()?;
            }
            // self.rt.queue.finish()?;
            // let insert_ms = t_insert.elapsed().as_secs_f64() * 1000.0;

            // average_table
            // let t_avg = Instant::now();
            self.kernel_average
                .set_default_global_work_size(ocl::SpatialDims::One(gws_table));
            self.kernel_average
                .set_default_local_work_size(ocl::SpatialDims::One(self.lws));
            self.kernel_average.set_arg(0, d_centroids)?;
            self.kernel_average.set_arg(1, d_counts)?;
            self.kernel_average.set_arg(2, &(table_size as i32))?;
            unsafe {
                self.kernel_average.enq()?;
            }
            // self.rt.queue.finish()?;
            // let avg_ms = t_avg.elapsed().as_secs_f64() * 1000.0;

            // compact_voxels
            // let t_compact = Instant::now();
            self.kernel_compact
                .set_default_global_work_size(ocl::SpatialDims::One(gws_table));
            self.kernel_compact
                .set_default_local_work_size(ocl::SpatialDims::One(self.lws));

            self.kernel_compact.set_arg(0, d_keys)?;
            self.kernel_compact.set_arg(1, d_centroids)?;
            self.kernel_compact.set_arg(2, d_counts)?;
            self.kernel_compact.set_arg(3, d_remap)?;
            self.kernel_compact.set_arg(4, &(table_size as i32))?;
            self.kernel_compact.set_arg(5, d_output)?;
            self.kernel_compact.set_arg(6, d_counter)?;
            unsafe {
                self.kernel_compact.enq()?;
            }
            self.rt.queue.finish()?;
            // let elapsed_kernel = start.elapsed().as_secs_f64() * 1000.0;

            // DtoH
            let t_dtoh = Instant::now();
            let mut host_count = [0i32; 1];
            d_counter
                .read(host_count.as_mut_slice())
                .enq()
                .context("Failed to read valid count")?;
            self.rt.queue.finish()?;
            let dtoh_ms = t_dtoh.elapsed().as_secs_f64() * 1000.0;

            let valid_count = host_count[0].max(0) as usize;

            // 結果出力
            // println!("=== GPU Voxel Downsample Timing ===");
            // println!("HtoD transfer:     {:.3} ms", htod_ms);
            // // println!("init_table:        {:.3} ms", init_ms);
            // // println!("insert_points:     {:.3} ms", insert_ms);
            // // println!("average_table:     {:.3} ms", avg_ms);
            // // println!("compact_voxels:    {:.3} ms", compact_ms);
            // // println!("DtoH transfer:     {:.3} ms", dtoh_ms);
            // println!("Total kernel:      {:.3} ms", elapsed_kernel);
            // println!("Total (with xfer): {:.3} ms", htod_ms + elapsed_kernel + dtoh_ms);
            // println!("===================================");

            // Create a new buffer and copy the result to avoid overwriting on next call
            let result_buffer = Buffer::<f32>::builder()
                .queue(self.rt.queue.clone())
                .flags(MemFlags::new().read_write())
                .len(valid_count * 3)
                .build()
                .context("Failed to create result buffer")?;

            // Copy only the valid data (valid_count * 3 elements)
            d_output
                .cmd()
                .copy(&result_buffer, Some(0), Some(valid_count * 3))
                .enq()
                .context("Failed to copy output buffer")?;

            self.rt.queue.finish()?;

            Ok((result_buffer, valid_count))
        }
    }
}
