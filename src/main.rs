// Some code is from: 2016 The vulkano developers
// Licensed under the MIT http://opensource.org/licenses/MIT.

use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
use vulkano::image::SwapchainImage;
use vulkano::image::attachment::AttachmentImage;
use vulkano::instance::Instance;
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::vertex::TwoBuffersDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError, ColorSpace, FullscreenExclusive};
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

#[derive(Default, Copy, Clone)]
pub struct Vertex {
    position: (f32, f32, f32)
}

#[derive(Default, Copy, Clone)]
pub struct Normal {
    normal: (f32, f32, f32)
}

vulkano::impl_vertex!(Normal, normal);

vulkano::impl_vertex!(Vertex, position);

type Point = (f32, f32, f32);
type Face = (u16, u16, u16);

// Inspired by http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html
struct Icosphere {
    index: u16,
    vertex_buffer: Vec<Vertex>,
    normal_buffer: Vec<Normal>,
    index_buffer: Vec<u16>,
    middle_vertex: HashMap<u32, u16>,

    radius: f32,
    tessellation: u8
}

impl Icosphere {
    fn new(radius: f32, tessellation: u8) -> Self {
        let mut is = Icosphere {
            index: 0,
            vertex_buffer: Vec::new(),
            normal_buffer: Vec::new(),
            index_buffer: Vec::new(),
            middle_vertex: HashMap::new(),

            radius: radius,
            tessellation: tessellation
        };

        // First, create a icosahedron
        // icosahedron can be constructed from 3 orthogonal rectangles. This is
        // the lenght of the side (golden ratio). The other side is 2 * 1.
        let t = (1.0 + 5.0_f32.sqrt()) / 2.0;

        // Rectangle no.1
        is.addVertex((-1.0,  t, 0.0));
        is.addVertex(( 1.0,  t, 0.0));
        is.addVertex((-1.0, -t, 0.0));
        is.addVertex(( 1.0, -t, 0.0));

        // Rectangle no.2
        is.addVertex((0.0, -1.0,  t));
        is.addVertex((0.0,  1.0,  t));
        is.addVertex((0.0, -1.0, -t));
        is.addVertex((0.0,  1.0, -t));

        // Rectangle no.3
        is.addVertex(( t, 0.0, -1.0));
        is.addVertex(( t, 0.0,  1.0));
        is.addVertex((-t, 0.0, -1.0));
        is.addVertex((-t, 0.0,  1.0));

        // Create a vector for holding the faces
        let mut faces: Vec<Face> = Vec::new();

        faces.push((0, 11, 5));
        faces.push((0, 5, 1));
        faces.push((0, 1, 7));
        faces.push((0, 7, 10));
        faces.push((0, 10, 11));

        faces.push((1, 5, 9));
        faces.push((5, 11, 4));
        faces.push((11, 10, 2));
        faces.push((10, 7, 6));
        faces.push((7, 1, 8));

        faces.push((3, 9, 4));
        faces.push((3, 4, 2));
        faces.push((3, 2, 6));
        faces.push((3, 6, 8));
        faces.push((3, 8, 9));

        faces.push((4, 9, 5));
        faces.push((2, 4, 11));
        faces.push((6, 2, 10));
        faces.push((8, 6, 7));
        faces.push((9, 8, 1));

        // Tessellate the icosahedron to create icosphere
        for _i in 0..is.tessellation {
            let mut newfaces: Vec<Face> = Vec::new();

            faces.iter().for_each(|x| {
                // Triangulate the triangle
                let newv1 = is.getMiddlePoint(x.0, x.1);
                let newv2 = is.getMiddlePoint(x.1, x.2);
                let newv3 = is.getMiddlePoint(x.2, x.0);

                newfaces.push((x.0, newv1, newv3));
                newfaces.push((x.1, newv2, newv1));
                newfaces.push((x.2, newv3, newv2));
                newfaces.push((newv1, newv2, newv3));
            });

            faces = newfaces;
        }

        // Create index buffer
        faces.iter().for_each(|x| {
            is.index_buffer.push(x.0);
            is.index_buffer.push(x.1);
            is.index_buffer.push(x.2);
        });

        is
    }

    fn addVertex(&mut self, point: Point) -> u16 {
        // normalize
        let rad = self.radius;
        let pointarray = [point.0, point.1, point.2];
        let amplitude = pointarray.iter().fold(0.0, |acc, x| acc + x.powi(2)).sqrt();
        let mut normalized = pointarray.iter().map(|x| x / amplitude);

        // Create a normal
        let normal = (normalized.next().unwrap(), normalized.next().unwrap(), normalized.next().unwrap());
        self.normal_buffer.push(Normal {
            normal: normal
        });

        // Add to vertex_buffer
        self.vertex_buffer.push(Vertex {
            position: (normal.0 * rad, normal.1 * rad, normal.2 * rad)
        });

        self.index += 1;
        self.index - 1
    }

    // This function gets the index of a point between two other points
    fn getMiddlePoint(&mut self, p1: u16, p2: u16) -> u16 {

        // Check for the middle point existance
        let sindex = cmp::min(p1, p2);
        let lindex = cmp::max(p1, p2);
        let key = (u32::from(sindex) << 16) | u32::from(lindex);

        // It exists
        if self.middle_vertex.contains_key(&key) {
            *self.middle_vertex.get(&key).unwrap()
        }

        // The middle point does not exist, create it
        else {
            let v1 = self.vertex_buffer[usize::from(p1)];
            let v2 = self.vertex_buffer[usize::from(p2)];
            
            let newindex = self.addVertex((
                (v1.position.0 + v2.position.0) / 2.0, 
                (v1.position.1 + v2.position.1) / 2.0, 
                (v1.position.2 + v2.position.2) / 2.0
            ));

            // Add it to the cache
            self.middle_vertex.insert(key, newindex);
            
            newindex
        }
    }

    fn getBuffers(self, device: Arc<Device>) -> (Arc<CpuAccessibleBuffer<[Vertex]>>, Arc<CpuAccessibleBuffer<[Normal]>>, Arc<CpuAccessibleBuffer<[u16]>>) {
        let vbuf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, self.vertex_buffer.iter().cloned()).unwrap();
        let nbuf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, self.normal_buffer.iter().cloned()).unwrap();
        let ibuf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, self.index_buffer.iter().cloned()).unwrap();

        (vbuf, nbuf, ibuf)
    }
}

fn main() {
    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, &required_extensions, None).unwrap();
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&event_loop, instance.clone()).unwrap();
    let dimensions: [u32; 2] = surface.window().inner_size().into();

    let queue_family = physical.queue_families().find(|&q|
        q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
    ).unwrap();

    let device_ext = DeviceExtensions { khr_swapchain: true, .. DeviceExtensions::none() };

    let (device, mut queues) = Device::new(
        physical, physical.supported_features(), &device_ext, [(queue_family, 0.5)].iter().cloned()
    ).unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();
        let usage = caps.supported_usage_flags;
        let format = caps.supported_formats[0].0;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        Swapchain::new(device.clone(), surface.clone(), caps.min_image_count, format, dimensions, 1,
            usage, &queue, SurfaceTransform::Identity, alpha, PresentMode::Fifo,
            FullscreenExclusive::Default, true, ColorSpace::SrgbNonLinear).unwrap()
    };

    let icos = Icosphere::new(0.5, 2);

    let buffers = icos.getBuffers(device.clone());
    let vertex_buffer = buffers.0;
    let normals_buffer = buffers.1;
    let index_buffer = buffers.2;

    let uniform_buffer = CpuBufferPool::<vs::ty::Data>::new(device.clone(), BufferUsage::all());

    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    let render_pass = Arc::new(
        vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
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

    let (mut pipeline, mut framebuffers) = window_size_dependent_setup(device.clone(), &vs, &fs, &images, render_pass.clone());
    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);
    let rotation_start = Instant::now();

    event_loop.run(move |event, _, control_flow| {
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
                    let (new_pipeline, new_framebuffers) = window_size_dependent_setup(device.clone(), &vs, &fs, &new_images, render_pass.clone());
                    pipeline = new_pipeline;
                    framebuffers = new_framebuffers;
                    recreate_swapchain = false;
                }

                let uniform_buffer_subbuffer = {
                    let elapsed = rotation_start.elapsed();
                    let rotation = elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
                    let rotation = Matrix3::from_angle_y(Rad(rotation as f32));

                    let aspect_ratio = dimensions[0] as f32 / dimensions[1] as f32;
                    let proj = cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.01, 100.0);
                    let view = Matrix4::look_at(Point3::new(0.3, 0.3, 1.0), Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
                    let scale = Matrix4::from_scale(1.0);

                    let uniform_data = vs::ty::Data {
                        world: Matrix4::from(rotation).into(),
                        view: (view * scale).into(),
                        proj: proj.into(),
                    };

                    uniform_buffer.next(uniform_data).unwrap()
                };

                let layout = pipeline.descriptor_set_layout(0).unwrap();
                let set = Arc::new(PersistentDescriptorSet::start(layout.clone())
                    .add_buffer(uniform_buffer_subbuffer).unwrap()
                    .build().unwrap()
                );

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

                let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
                    .begin_render_pass(
                        framebuffers[image_num].clone(), false,
                        vec![
                            [0.0, 0.0, 1.0, 1.0].into(),
                            1f32.into()
                        ]
                    ).unwrap()
                    .draw_indexed(
                        pipeline.clone(),
                        &DynamicState::none(),
                        vec!(vertex_buffer.clone(), normals_buffer.clone()),
                        index_buffer.clone(), set.clone(), ()).unwrap()
                    .end_render_pass().unwrap()
                    .build().unwrap();

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

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    device: Arc<Device>,
    vs: &vs::Shader,
    fs: &fs::Shader,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
) -> (Arc<dyn GraphicsPipelineAbstract + Send + Sync>, Vec<Arc<dyn FramebufferAbstract + Send + Sync>> ) {
    let dimensions = images[0].dimensions();

    let depth_buffer = AttachmentImage::transient(device.clone(), dimensions, Format::D16Unorm).unwrap();

    let framebuffers = images.iter().map(|image| {
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone()).unwrap()
                .add(depth_buffer.clone()).unwrap()
                .build().unwrap()
        ) as Arc<dyn FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>();

    let pipeline = Arc::new(GraphicsPipeline::start()
        .vertex_input(TwoBuffersDefinition::<Vertex, Normal>::new())
        .vertex_shader(vs.main_entry_point(), ())
        .triangle_list()
        .viewports_dynamic_scissors_irrelevant(1)
        .viewports(iter::once(Viewport {
            origin: [0.0, 0.0],
            dimensions: [dimensions[0] as f32, dimensions[1] as f32],
            depth_range: 0.0 .. 1.0,
        }))
        .fragment_shader(fs.main_entry_point(), ())
        .depth_stencil_simple_depth()
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap());

    (pipeline, framebuffers)
}

mod vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: "                
            #version 450

            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 normal;

            layout(location = 0) out vec3 v_normal;

            layout(set = 0, binding = 0) uniform Data {
                mat4 world;
                mat4 view;
                mat4 proj;
            } uniforms;

            void main() {
                mat4 worldview = uniforms.view * uniforms.world;
                v_normal = transpose(inverse(mat3(worldview))) * normal;
                gl_Position = uniforms.proj * worldview * vec4(position, 1.0);
            }
        "
    }
}

mod fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "
            #version 450

            layout(location = 0) in vec3 v_normal;
            layout(location = 0) out vec4 f_color;

            const vec3 LIGHT = vec3(0.0, 0.0, 1.0);

            void main() {
                float brightness = min(dot(normalize(v_normal), normalize(LIGHT)), dot(-normalize(v_normal), normalize(LIGHT)));
                vec3 dark_color = vec3(0.6, 0.0, 0.0);
                vec3 regular_color = vec3(1.0, 0.0, 0.0);

                f_color = vec4(mix(dark_color, regular_color, brightness), 1.0);
            }
        "
    }
}