
use cgmath::{Vector3, Point3, Matrix3, Rad};
use winit::event::{Event, WindowEvent, KeyboardInput, VirtualKeyCode, ElementState};

struct MoveStatus {
    up: f32,
    down: f32,
    left: f32,
    right: f32,
    forward: f32,
    back: f32,
    pitch_up: f32,
    pitch_down: f32,
    yaw_left: f32,
    yaw_right: f32,
    roll_left: f32,
    roll_right: f32
}

pub struct Camera {
    movement_speed: f32,
    roll_speed: f32,

    position: Vector3<f32>,
    rotation: Vector3<f32>,

    mouse_status: bool,
    move_status: MoveStatus,
    move_vector: Vector3<f32>,
    rotation_vector: Vector3<f32>,

    screenx: u32,
    screeny: u32,
}

impl Camera {
    pub fn new(pos: Vector3<f32>, x: u32, y: u32) -> Self {
        Self {
            movement_speed: 6.0,
            roll_speed: 2.0,
            position: pos,
            rotation: Vector3::new(0.0, 0.0, 0.0),
            mouse_status: false,
            move_status: MoveStatus {
                up: 0.0,
                down: 0.0,
                left: 0.0,
                right: 0.0,
                forward: 0.0,
                back: 0.0,
                pitch_up: 0.0,
                pitch_down: 0.0,
                yaw_left: 0.0,
                yaw_right: 0.0,
                roll_left: 0.0,
                roll_right: 0.0
            },
            move_vector: Vector3::new(0.0, 0.0, 0.0),
            rotation_vector: Vector3::new(0.0, 0.0, 0.0),
            screenx: x,
            screeny: y,
        }
    }

    pub fn key_update(&mut self, key_code: VirtualKeyCode, press_status: bool) {
        let press_status: f32 = if press_status {1.0} else {0.0};

        match key_code {
            VirtualKeyCode::W => self.move_status.forward = press_status,
            VirtualKeyCode::S => self.move_status.back = press_status,
            VirtualKeyCode::A => self.move_status.left = press_status,
            VirtualKeyCode::D => self.move_status.right = press_status,
            VirtualKeyCode::LShift => self.move_status.up = press_status,
            VirtualKeyCode::Space => self.move_status.down = press_status,
            VirtualKeyCode::Down => self.move_status.pitch_up = press_status,
            VirtualKeyCode::Up => self.move_status.pitch_down = press_status,
            VirtualKeyCode::Left => self.move_status.yaw_left = press_status,
            VirtualKeyCode::Right => self.move_status.yaw_right = press_status,
            VirtualKeyCode::E => self.move_status.roll_left = press_status,
            VirtualKeyCode::Q => self.move_status.roll_right = press_status,
            _ => {}
        }

        self.updata_move_vector();
        self.update_rotation_vector();
    }

    pub fn mouse_move(&mut self, dx: i32, dy: i32) {

        // let halfx = self.screenx / 2;
        // let halfy = self.screeny / 2;

        // if dx > 0 {
        //     self.move_status.yaw_left = dx as f32 / 2.0;
        // }

        // else if dx < 0 {
        //     self.move_status.yaw_right = -dx as f32 / 2.0;
        // }

        // else {
        //     self.move_status.yaw_left = 0.0;
        //     self.move_status.yaw_right = 0.0;
        // }

        // if dy < 0 {
        //     self.move_status.pitch_up = -dy as f32 / 2.0;
        // }

        // else if dy > 0 {
        //     self.move_status.pitch_down = dy as f32 / 2.0;
        // }

        // else {
        //     self.move_status.pitch_up = 0.0;
        //     self.move_status.pitch_down = 0.0;
        // }

        // self.update_rotation_vector();
    }

    pub fn update(&mut self, dt: f32) {
        let move_mult = dt * self.movement_speed;
        let roll_mult = dt * self.roll_speed;

        // Get forward vector
        let forward = self.get_forward();

        // Get up vector
        let up = self.get_up();

        // Get right vector
        let right = self.get_right();

        // Translation
        // In x axis
        if (self.move_vector.x != 0.0) {
            self.position += move_mult * right * self.move_vector.x;
        }

        // In y axis
        if (self.move_vector.y != 0.0) {
            self.position += move_mult * up * self.move_vector.y;
        }

        // In z axis
        if (self.move_vector.z != 0.0) {
            self.position += move_mult * forward * self.move_vector.z;
        }

        // Rotation:
        self.rotation += roll_mult * self.rotation_vector;

        // Clear the mouse movement
        self.mouse_move(0, 0);

    }

    fn updata_move_vector(&mut self) {
        self.move_vector.x = -self.move_status.left + self.move_status.right;
        self.move_vector.y = -self.move_status.down + self.move_status.up;
        self.move_vector.z = -self.move_status.forward + self.move_status.back;
    }

    fn update_rotation_vector(&mut self) {
        self.rotation_vector.x = -self.move_status.pitch_down + self.move_status.pitch_up;
        self.rotation_vector.y = -self.move_status.yaw_right + self.move_status.yaw_left;
        self.rotation_vector.z = -self.move_status.roll_right + self.move_status.roll_left;
    }

    pub fn get_position(&self) -> Point3<f32> {
        Point3::new(self.position.x as f32, self.position.y as f32, self.position.z as f32)
    }

    fn get_rotmat(&self) -> Matrix3<f32> {
        Matrix3::from_angle_x(Rad(self.rotation.x)) * Matrix3::from_angle_y(Rad(self.rotation.y)) * Matrix3::from_angle_z(Rad(self.rotation.z))
    }

    pub fn get_forward(&self) -> Vector3<f32> {
        let rotation = self.get_rotmat();
        (rotation * Vector3::<f32>::new(0.0, 0.0, 1.0))
    }

    pub fn get_up(&self) -> Vector3<f32> {
        let rotation = self.get_rotmat();
        rotation * Vector3::<f32>::new(0.0, 1.0, 0.0)
    }

    pub fn get_right(&self) -> Vector3<f32> {
        let rotation = self.get_rotmat();
        rotation * Vector3::<f32>::new(1.0, 0.0, 0.0)
    }

    pub fn get_lookat(&self) -> Point3<f32> {
        let lookat_point = -self.get_forward() + self.position;
        Point3::new(lookat_point.x, lookat_point.y, lookat_point.z)
    }
}