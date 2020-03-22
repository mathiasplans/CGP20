use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState, CommandBuffer, AutoCommandBuffer};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions, QueuesIter, Queue};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
use vulkano::image::SwapchainImage;
use vulkano::image::attachment::AttachmentImage;
use vulkano::instance::Instance;
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::vertex::TwoBuffersDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError, ColorSpace, FullscreenExclusive, Surface};
use vulkano::swapchain;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::sync;

use vulkano_win::VkSurfaceBuild;
use winit::window::{WindowBuilder, Window};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::event::{Event, WindowEvent};

use cgmath::{Matrix3, Matrix4, Point3, Vector3, Rad};

use std::iter;
use std::sync::Arc;
use std::time::Instant;

use std::collections::HashMap;
use std::cmp;

// #[path = "icosphere.rs"] mod icosphere;
// use icosphere::Icosphere;
use crate::icosphere::Icosphere;

pub struct Renderer {
    surface: Arc<Surface<Window>>,
    queues: QueuesIter,

    swapchain: Arc<Swapchain<Window>>,
    images: Vec<Arc<SwapchainImage<Window>>>,

    dimensions: [u32; 2],

    device: Arc<Device>,
    queue: Arc<Queue>,
    event_loop: EventLoop<()>,

    objects: Vec<Icosphere>
}

impl Renderer {
    pub fn setup() -> Self {
        let required_extensions = vulkano_win::required_extensions();
        let instance = Instance::new(None, &required_extensions, None).unwrap();
        let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
        println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

        let event_loop = EventLoop::new();
        let surface = WindowBuilder::new().build_vk_surface(&event_loop, instance.clone()).unwrap();
        let dimensions: [u32; 2] = surface.window().inner_size().into();
        println!("{:?}", dimensions);

        let queue_family = physical.queue_families().find(|&q|
            q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
        ).unwrap();

        let device_ext = DeviceExtensions { khr_swapchain: true, .. DeviceExtensions::none() };

        let mut dev = Device::new(
            physical, physical.supported_features(), &device_ext, [(queue_family, 0.5)].iter().cloned()
        ).unwrap();

        let queue = dev.1.next().unwrap();

        let (swapchain, images) = {
            let caps = surface.capabilities(physical).unwrap();
            let usage = caps.supported_usage_flags;
            let format = caps.supported_formats[0].0;
            let alpha = caps.supported_composite_alpha.iter().next().unwrap();

            Swapchain::new(dev.0.clone(), surface.clone(), caps.min_image_count, format, dimensions, 1,
                usage, &queue, SurfaceTransform::Identity, alpha, PresentMode::Fifo,
                FullscreenExclusive::Default, true, ColorSpace::SrgbNonLinear).unwrap()
        };

        Renderer {
            surface: surface,
            queues: dev.1,

            swapchain: swapchain,
            images: images,

            dimensions: dimensions,

            device: dev.0,
            queue: queue,
            event_loop: event_loop,

            objects: Vec::new()
        }
    }

    pub fn get_device(&self) -> Arc<Device> {
        self.device.clone()
    }

    pub fn start(self, objects: Vec<Icosphere>) {
        let render_pass = Arc::new(
            vulkano::single_pass_renderpass!(self.device.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: Store,
                        format: self.swapchain.format(),
                        samples: 1,
                    },
                    depth: {
                        load: Clear,
                        store: DontCare,
                        format: Format::D16Unorm,
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {depth}
                }
            ).unwrap()
        );

        let mut framebuffers = {
            let dimensions = self.images[0].dimensions();

            let depth_buffer = AttachmentImage::transient(self.device.clone(), dimensions, Format::D16Unorm).unwrap();

            let framebuffers = self.images.iter().map(|image| {
                Arc::new(
                    Framebuffer::start(render_pass.clone())
                        .add(image.clone()).unwrap()
                        .add(depth_buffer.clone()).unwrap()
                        .build().unwrap()
                ) as Arc<dyn FramebufferAbstract + Send + Sync>
            }).collect::<Vec<_>>();

            framebuffers
        };

        let mut recreate_swapchain = false;

        let mut previous_frame_end = Some(Box::new(sync::now(self.device.clone())) as Box<dyn GpuFuture>);

        let mut swapchain = self.swapchain.clone();
        let device = self.device.clone();
        let surface = self.surface.clone();
        let queue = self.queue.clone();

        self.event_loop.run(move |event, _, control_flow| {
            match event {
                Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                    *control_flow = ControlFlow::Exit;
                },
                Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
                    recreate_swapchain = true;
                },
                Event::RedrawEventsCleared => {
                    previous_frame_end.as_mut().unwrap().cleanup_finished();

                    if recreate_swapchain {
                        let dimensions: [u32; 2] = surface.window().inner_size().into();
                        let (new_swapchain, new_images) = match swapchain.recreate_with_dimensions(dimensions) {
                            Ok(r) => r,
                            Err(SwapchainCreationError::UnsupportedDimensions) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e)
                        };

                        swapchain = new_swapchain;
                        framebuffers = {
                            let dimensions = new_images[0].dimensions();

                            let depth_buffer = AttachmentImage::transient(device.clone(), dimensions, Format::D16Unorm).unwrap();

                            let framebuffers = new_images.iter().map(|image| {
                                Arc::new(
                                    Framebuffer::start(render_pass.clone())
                                        .add(image.clone()).unwrap()
                                        .add(depth_buffer.clone()).unwrap()
                                        .build().unwrap()
                                ) as Arc<dyn FramebufferAbstract + Send + Sync>
                            }).collect::<Vec<_>>();

                            framebuffers
                        };
                        recreate_swapchain = false;
                    }

                    let (image_num, suboptimal, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        },
                        Err(e) => panic!("Failed to acquire next image: {:?}", e)
                    };

                    if suboptimal {
                        recreate_swapchain = true;
                    }

                    let mut command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
                        .begin_render_pass(
                            framebuffers[image_num].clone(), false,
                            vec![
                                [0.0, 0.0, 1.0, 1.0].into(),
                                1f32.into()
                            ]
                        ).unwrap();

                    for x in &objects {
                        let buffer = x.get_buffers();
                        let pipeline = x.get_pipeline(device.clone(), render_pass.clone());
                        let descriptor_set = x.get_uniforms((*pipeline.descriptor_set_layout(0).unwrap()).clone());

                        command_buffer = command_buffer.draw_indexed(
                            pipeline.clone(),
                            &DynamicState::none(),
                            vec!(buffer.0.clone(), buffer.1.clone()),
                            buffer.2.clone(), descriptor_set.clone(), ()
                        ).unwrap();
                    }

                    let command_buffer = command_buffer.end_render_pass().unwrap().build().unwrap();

                    let future = previous_frame_end.take().unwrap()
                        .join(acquire_future)
                        .then_execute(queue.clone(), command_buffer).unwrap()
                        .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                        .then_signal_fence_and_flush();

                    match future {
                        Ok(future) => {
                            previous_frame_end = Some(Box::new(future) as Box<_>);
                        },
                        Err(FlushError::OutOfDate) => {
                            recreate_swapchain = true;
                            previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                        }
                        Err(e) => {
                            println!("Failed to flush future: {:?}", e);
                            previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                        }
                    }
                },
                _ => ()
            }
        });
    }
}