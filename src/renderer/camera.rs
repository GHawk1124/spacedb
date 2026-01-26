//! Camera controller for 3D viewport

use glam::{Mat4, Vec3};

/// Orbital camera that rotates around a target point
#[derive(Debug, Clone)]
pub struct Camera {
    /// Target point the camera looks at (usually Earth center)
    pub target: Vec3,
    /// Distance from target
    pub distance: f32,
    /// Azimuth angle (rotation around Y axis) in radians
    pub azimuth: f32,
    /// Elevation angle (rotation above/below XZ plane) in radians
    pub elevation: f32,
    /// Field of view in radians
    pub fov: f32,
    /// Near clip plane
    pub near: f32,
    /// Far clip plane
    pub far: f32,
    /// Minimum zoom distance
    pub min_distance: f32,
    /// Maximum zoom distance
    pub max_distance: f32,
    /// Default distance from target
    pub default_distance: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 4.0, // About 4 Earth radii out
            azimuth: 0.0,
            elevation: 0.3, // Slightly above equator
            fov: 45.0_f32.to_radians(),
            near: 0.01,
            far: 100.0,
            min_distance: 1.1,
            max_distance: 50.0,
            default_distance: 4.0,
        }
    }
}

impl Camera {
    /// Get camera position in world space
    pub fn position(&self) -> Vec3 {
        let x = self.distance * self.elevation.cos() * self.azimuth.sin();
        let y = self.distance * self.elevation.sin();
        let z = self.distance * self.elevation.cos() * self.azimuth.cos();
        self.target + Vec3::new(x, y, z)
    }

    /// Get view matrix
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position(), self.target, Vec3::Y)
    }

    /// Get projection matrix
    pub fn projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov, aspect_ratio, self.near, self.far)
    }

    /// Get combined view-projection matrix
    pub fn view_projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
        self.projection_matrix(aspect_ratio) * self.view_matrix()
    }

    /// Orbit the camera (mouse drag)
    pub fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        self.azimuth += delta_x * 0.01;
        self.elevation = (self.elevation + delta_y * 0.01).clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );
    }

    /// Zoom the camera (mouse wheel)
    pub fn zoom(&mut self, delta: f32) {
        self.distance =
            (self.distance * (1.0 - delta * 0.1)).clamp(self.min_distance, self.max_distance);
    }

    /// Pan the camera (shift + mouse drag)
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let right = Vec3::new(self.azimuth.cos(), 0.0, -self.azimuth.sin());
        let up = Vec3::Y;
        self.target += right * delta_x * 0.01 * self.distance;
        self.target += up * delta_y * 0.01 * self.distance;
    }

    /// Focus on a point at a given altitude (in Earth radii)
    pub fn focus_on_altitude(&mut self, position: Vec3, altitude_er: f32) {
        // Position camera to view the satellite with Earth in background
        self.enable_close_zoom(true);
        self.target = position;
        self.distance = (altitude_er * 0.5).max(self.min_distance);
    }

    pub fn enable_close_zoom(&mut self, enabled: bool) {
        if enabled {
            self.min_distance = 0.05;
            self.near = 0.001;
        } else {
            self.min_distance = 1.1;
            self.near = 0.01;
            if self.distance < self.default_distance {
                self.distance = self.default_distance;
            }
        }
    }

    pub fn reset_to_earth(&mut self) {
        self.target = Vec3::ZERO;
        self.enable_close_zoom(false);
    }
}

/// Camera uniform data for shaders
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
}

impl CameraUniform {
    pub fn from_camera(camera: &Camera, aspect_ratio: f32) -> Self {
        let pos = camera.position();
        Self {
            view_proj: camera
                .view_projection_matrix(aspect_ratio)
                .to_cols_array_2d(),
            view: camera.view_matrix().to_cols_array_2d(),
            proj: camera.projection_matrix(aspect_ratio).to_cols_array_2d(),
            camera_pos: [pos.x, pos.y, pos.z, 1.0],
        }
    }
}
