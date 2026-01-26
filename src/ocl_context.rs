use anyhow::{Context as AnyhowContext, Result};
use ocl::{Context, Device, Platform, Queue};

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
