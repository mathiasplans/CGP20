use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState, CommandBuffer, AutoCommandBuffer};
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

use vulkano::pipeline::ComputePipeline;

use vulkano::descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSetBuf;
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::descriptor::descriptor_set::{PersistentDescriptorSet, PersistentDescriptorSetBuilder};

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

        let device_ext = DeviceExtensions {
            khr_swapchain: true,
            khr_storage_buffer_storage_class: true,
            .. DeviceExtensions::none() };

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

    pub fn get_planet_buffer(&self, objects: &'static Vec<Icosphere>) -> Arc<CpuAccessibleBuffer<[movement_cs::ty::planet_struct]>> {
        unsafe {
            // CpuAccessibleBuffer::uninitialized_array(self.get_device(), planet_amount as usize, BufferUsage::all(), false).unwrap()
            CpuAccessibleBuffer::from_iter(
                self.device.clone(),
                BufferUsage::all(),
                false,
                objects.into_iter().map(|x| {
                    print!("{:?}\n", x.get_position());
                    movement_cs::ty::planet_struct {
                        _dummy0: [0, 0, 0, 0],
                        _dummy1: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        pos: x.get_position(),
                        velocity: [0.0, 0.0, 0.0],
                        mass: x.get_mass(),
                        rad: x.get_rad()
                    }
                })
            ).unwrap()
        }
    }

    pub fn start(self, objects: &'static Vec<Icosphere>) {
        // Get the amount of planets
        let planet_amount = objects.len() as u32;
        print!("Total amount of planets: {:?}\n", planet_amount);

        // Create a buffer for planet info
        // Specify the type.
        let planet_buffer = self.get_planet_buffer(objects);

        // // Specialization constants for the gravity shader
        // let spec_const = movement_cs::SpecializationConstants {
        //     local_size_x_id: planet_amount
        // };
                
        // Create a compute shader for movement
        let movement_shader = movement_cs::Shader::load(self.get_device())
            .expect("failed to create shader module");

        // Create a compute pipeline for movement
        let movement_pipeline = Arc::new(ComputePipeline::new(self.get_device(), &movement_shader.main_entry_point(), &(/*spec_const*/))
            .expect("failed to create compute pipeline"));

        // Create a descriptor set for gravity shader
        let movement_set = Arc::new(PersistentDescriptorSet::start(movement_pipeline.clone().descriptor_set_layout(0).unwrap().clone())
            .add_buffer(planet_buffer.clone()).unwrap()
            .build().unwrap()
        );

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

        let mut compute_pc = movement_cs::ty::PushConstantData {
            time: 0.0,
            delta: 0.0,
            circular: 0.0
        };

        let compute_time = Instant::now();

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

                    let elapsed = compute_time.elapsed();
                    let last_time = compute_pc.time;
                    compute_pc.time = elapsed.as_secs() as f32 + elapsed.subsec_nanos() as f32 / 1_000_000_000.0;
                    compute_pc.delta = compute_pc.time - last_time;
                    compute_pc.circular = compute_pc.time;

                    // print!("{:?}\n", compute_pc.delta);

                    if compute_pc.circular > 6.28318530718 {
                        compute_pc.circular -= 6.28318530718;
                    }

                    let mut command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
                        // Compute pipeline
                        // Movement
                        .dispatch([planet_amount, 1, 1], movement_pipeline.clone(), movement_set.clone(), compute_pc).unwrap()
                        
                        // Graphich pipeline
                        .begin_render_pass(
                            framebuffers[image_num].clone(), false,
                            vec![
                                [0.0, 0.0, 1.0, 1.0].into(),
                                1f32.into()
                            ]
                        ).unwrap();

                    for x in objects {
                        let buffer = x.get_buffers();
                        let pipeline = x.get_pipeline(device.clone(), render_pass.clone());
                        let uniforms = x.get_uniforms();


                        let descriptor_set = PersistentDescriptorSet::start(pipeline.descriptor_set_layout(0).unwrap().clone())
                            .add_buffer(uniforms).unwrap()
                            .add_buffer(planet_buffer.clone()).unwrap()
                            .build().unwrap();

                        command_buffer = command_buffer.draw_indexed(
                            pipeline.clone(),
                            &DynamicState::none(),
                            vec!(buffer.0.clone(), buffer.1.clone()),
                            buffer.2.clone(), descriptor_set, planet_amount
                        ).unwrap();
                    }

                    let command_buffer = command_buffer
                        .end_render_pass().unwrap()
                        .build().unwrap();

                    let future = previous_frame_end.take().unwrap()
                        .join(acquire_future)
                        .then_execute(queue.clone(), command_buffer).unwrap()
                        .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                        .then_signal_fence_and_flush();

                    // print!("{:?} {:?}\n", planet_buffer.read().unwrap()[0].pos, planet_buffer.read().unwrap()[1].pos);

                    match future  {
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

fn distanceb(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powf(2.0) + (a[1] - b[1]).powf(2.0) + (a[2] - b[2]).powf(2.0)).sqrt()
}

mod movement_cs {
    vulkano_shaders::shader!{
        ty: "compute",
        path: "src/shaders/movement.comp"
    }
}

#[allow(dead_code)]
const Y: &str = include_str!("shaders/movement.comp");