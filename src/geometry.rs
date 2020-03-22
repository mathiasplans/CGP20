pub type Point = (f32, f32, f32);
pub type Vector = Point;
pub type Face = (u16, u16, u16);

#[derive(Default, Copy, Clone)]
pub struct Vertex {
    pub position: Point
}

#[derive(Default, Copy, Clone)]
pub struct Normal {
    pub normal: Vector
}

vulkano::impl_vertex!(Vertex, position);
vulkano::impl_vertex!(Normal, normal);

