use std::ops;

pub struct RGBA {
    r: f32,
    g: f32,
    b: f32,
    a: f32
}

impl RGBA {
    pub fn new(color_code: u32, mut alpha: f32) -> Self {

        if alpha > 1.0 {
            alpha = 1.0;
        }

        else if alpha < 0.0 {
            alpha = 0.0;
        }

        Self {
            r: ((color_code & 0xFF0000) >> 16) as f32 / 256.0,
            g: ((color_code & 0x00FF00) >> 8) as f32 / 256.0,
            b: (color_code & 0x0000FF) as f32 / 256.0,
            a: alpha
        }
    }

    pub fn as_rgb(&self) -> [f32; 3] {
        [self.r, self.g, self.b]
    }

    pub fn as_rgba(&self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }

    pub fn as_rg(&self) -> [f32; 2] {
        [self.r, self.g]
    }

    pub fn as_rb(&self) -> [f32; 2] {
        [self.r, self.g]
    }

    pub fn as_gb(&self) -> [f32; 2] {
        [self.g, self.b]
    }

    pub fn as_r(&self) -> f32 {
        self.r
    }

    pub fn as_g(&self) -> f32 {
        self.g
    }

    pub fn as_b(&self) -> f32 {
        self.b
    }

    pub fn as_a(&self) -> f32 {
        self.a
    }

    pub fn set_r(&mut self, newr: f32) {
        self.r = newr;
    }

    pub fn set_g(&mut self, newg: f32) {
        self.g = newg;
    }

    pub fn set_b(&mut self, newb: f32) {
        self.b = newb;
    }

    pub fn set_a(&mut self, newa: f32) {
        self.a = newa;
    }
}

impl ops::Add for RGBA {
    type Output = Self;

    fn add(self, _rhs: Self) -> Self {
        Self {
            r: self.r + _rhs.r,
            g: self.g + _rhs.g,
            b: self.b + _rhs.b,
            a: self.a
        }
    }
}

impl ops::AddAssign for RGBA {
    fn add_assign(&mut self, _rhs: Self) {
        *self = Self {
            r: self.r + _rhs.r,
            g: self.g + _rhs.g,
            b: self.b + _rhs.b,
            a: self.a
        }
    }
}

impl ops::Sub for RGBA {
    type Output = Self;

    fn sub(self, _rhs: Self) -> Self {
        Self {
            r: self.r - _rhs.r,
            g: self.g - _rhs.g,
            b: self.b - _rhs.b,
            a: self.a
        }
    }
}

impl ops::SubAssign for RGBA {
    fn sub_assign(&mut self, _rhs: Self) {
        *self = Self {
            r: self.r - _rhs.r,
            g: self.g - _rhs.g,
            b: self.b - _rhs.b,
            a: self.a
        }
    }
}

impl ops::Mul<f32> for RGBA {
    type Output = Self;

    fn mul(self, _rhs: f32) -> Self {
        Self {
            r: self.r * _rhs,
            g: self.g * _rhs,
            b: self.b * _rhs,
            a: self.a
        }
    }
}

impl ops::MulAssign<f32> for RGBA {
    fn mul_assign(&mut self, _rhs: f32) {
        *self = Self {
            r: self.r * _rhs,
            g: self.g * _rhs,
            b: self.b * _rhs,
            a: self.a
        }
    }
}

impl ops::Div<f32> for RGBA {
    type Output = Self;

    fn div(self, _rhs: f32) -> Self {
        Self {
            r: self.r / _rhs,
            g: self.g / _rhs,
            b: self.b / _rhs,
            a: self.a
        }
    }
}

impl ops::DivAssign<f32> for RGBA {
    fn div_assign(&mut self, _rhs: f32) {
        *self = Self {
            r: self.r / _rhs,
            g: self.g / _rhs,
            b: self.b / _rhs,
            a: self.a
        }
    }
}