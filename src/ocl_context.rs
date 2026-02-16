use anyhow::{Context as AnyhowContext, Result};
use ocl::{Context, Device, Platform, Queue};

use crate::{
    gpu_cov::OclCovContext, gpu_gicp::OclGicpContext, gpu_icp::OclIcpContext,
    gpu_normals::OclNormalsContext, gpu_search::OclSearchContext,
    gpu_search_neighbors::OclSearchNeighborsContext, gpu_transform::OclTransformContext,
    gpu_voxel::OclVoxelContext,
};

#[derive(Clone)]
pub struct OclRuntime {
    pub context: Context,
    pub queue: Queue,
    pub device: Device,
    pub platform: Platform,
}

impl OclRuntime {
    pub fn new() -> Result<Self> {
        let platform = Platform::default();
        let device = Device::first(platform).context("Failed to get default device")?;
        let context = Context::builder()
            .platform(platform)
            .devices(device.clone())
            .build()
            .context("Failed to build OpenCL context")?;

        let queue =
            Queue::new(&context, device.clone(), None).context("Failed to create command queue")?;

        Ok(Self {
            context,
            queue,
            device,
            platform,
        })
    }
}

pub struct OclContexts<'a> {
    pub gpu_voxel: &'a mut OclVoxelContext,
    pub gpu_covs: &'a mut OclCovContext,
    pub gpu_normals: &'a mut OclNormalsContext,
    pub gpu_transform: &'a mut OclTransformContext,
    pub gpu_search: &'a mut OclSearchContext,
    pub gpu_search_neighbors: &'a mut OclSearchNeighborsContext,
    pub gpu_gicp: &'a mut OclGicpContext,
    pub gpu_icp: &'a mut OclIcpContext,
}
