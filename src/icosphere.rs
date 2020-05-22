
use std::sync::Arc;

use std::collections::HashMap;
use std::cmp;

use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::{PersistentDescriptorSet, PersistentDescriptorSetBuilder};
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

use vulkano::descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSetBuf;
use vulkano::buffer::cpu_pool::CpuBufferPoolSubbuffer;
use vulkano::memory::pool::StdMemoryPool;


use std::time::Instant;
use cgmath::{Matrix3, Matrix4, Point3, Vector3, Vector4, Rad};

use std::iter;

// #[path = "geometry.rs"] mod geometry;
use crate::geometry::{Vertex, Normal, Point, Vector, Face};

// Inspired by http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html
pub struct Icosphere {
    index: u16,
    vertex_vector: Vec<Vertex>,
    normal_vector: Vec<Normal>,
    index_vector: Vec<u16>,
    middle_vertex: HashMap<u32, u16>,

    radius: f32,
    tessellation: u8,
}

impl Icosphere {
    pub fn get_vectors(self) -> (Vec<Vertex>, Vec<Normal>, Vec<u16>) {
        (self.vertex_vector, self.normal_vector, self.index_vector)
    }

    pub fn new(radius: f32, tessellation: u8) -> Self {
        let mut is = Icosphere {
            index: 0,
            vertex_vector: Vec::new(),
            normal_vector: Vec::new(),
            index_vector: Vec::new(),
            middle_vertex: HashMap::new(),

            radius: radius,
            tessellation: tessellation
        };

        // First, create a icosahedron
        // icosahedron can be constructed from 3 orthogonal rectangles. This is
        // the lenght of the side (golden ratio). The other side is 2 * 1.
        let t = (1.0 + 5.0_f32.sqrt()) / 2.0;

        // Rectangle no.1
        is.add_vertex((-1.0,  t, 0.0));
        is.add_vertex(( 1.0,  t, 0.0));
        is.add_vertex((-1.0, -t, 0.0));
        is.add_vertex(( 1.0, -t, 0.0));

        // Rectangle no.2
        is.add_vertex((0.0, -1.0,  t));
        is.add_vertex((0.0,  1.0,  t));
        is.add_vertex((0.0, -1.0, -t));
        is.add_vertex((0.0,  1.0, -t));

        // Rectangle no.3
        is.add_vertex(( t, 0.0, -1.0));
        is.add_vertex(( t, 0.0,  1.0));
        is.add_vertex((-t, 0.0, -1.0));
        is.add_vertex((-t, 0.0,  1.0));

        // Create a vector for holding the faces
        let mut faces: Vec<Face> = Vec::new();

        faces.push((5, 11, 0));
        faces.push((1, 5, 0));
        faces.push((7, 1, 0));
        faces.push((10, 7, 0));
        faces.push((11, 10, 0));

        faces.push((9, 5, 1));
        faces.push((4, 11, 5));
        faces.push((2, 10, 11));
        faces.push((6, 7, 10));
        faces.push((8, 1, 7));

        faces.push((4, 9, 3));
        faces.push((2, 4, 3));
        faces.push((6, 2, 3));
        faces.push((8, 6, 3));
        faces.push((9, 8, 3));

        faces.push((5, 9, 4));
        faces.push((11, 4, 2));
        faces.push((10, 2, 6));
        faces.push((7, 6, 8));
        faces.push((1, 8, 9));

        // Tessellate the icosahedron to create icosphere
        for _i in 0..is.tessellation {
            let mut newfaces: Vec<Face> = Vec::new();

            faces.iter().for_each(|x| {
                // Triangulate the triangle
                let newv1 = is.get_middle_point(x.0, x.1);
                let newv2 = is.get_middle_point(x.1, x.2);
                let newv3 = is.get_middle_point(x.2, x.0);

                newfaces.push((x.0, newv1, newv3));
                newfaces.push((x.1, newv2, newv1));
                newfaces.push((x.2, newv3, newv2));
                newfaces.push((newv1, newv2, newv3));
            });

            faces = newfaces;
        }

        // Create index buffer
        faces.iter().for_each(|x| {
            is.index_vector.push(x.0);
            is.index_vector.push(x.1);
            is.index_vector.push(x.2);
        });

        is
    }

    fn add_vertex(&mut self, point: Point) -> u16 {
        // normalize
        let rad = self.radius;
        let pointarray = [point.0, point.1, point.2];
        let amplitude = pointarray.iter().fold(0.0, |acc, x| acc + x.powi(2)).sqrt();
        let mut normalized = pointarray.iter().map(|x| x / amplitude);

        // Create a normal
        let normal = (normalized.next().unwrap(), normalized.next().unwrap(), normalized.next().unwrap());
        self.normal_vector.push(Normal {
            normal: normal
        });

        // Add to vertex_buffer
        self.vertex_vector.push(Vertex {
            position: (normal.0 * rad, normal.1 * rad, normal.2 * rad)
        });

        self.index += 1;
        self.index - 1
    }

    // This function gets the index of a point between two other points
    fn get_middle_point(&mut self, p1: u16, p2: u16) -> u16 {

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
            let v1 = self.vertex_vector[usize::from(p1)];
            let v2 = self.vertex_vector[usize::from(p2)];

            let newindex = self.add_vertex((
                (v1.position.0 + v2.position.0) / 2.0,
                (v1.position.1 + v2.position.1) / 2.0,
                (v1.position.2 + v2.position.2) / 2.0
            ));

            // Add it to the cache
            self.middle_vertex.insert(key, newindex);

            newindex
        }
    }
}