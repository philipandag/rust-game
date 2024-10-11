use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::{FRect, FPoint};
use sdl2::render::{Canvas};
use sdl2::video::Window;
use core::{f32, f64};
use std::borrow::BorrowMut;
use std::cmp::Ordering;
use std::hash::Hash;
use std::ops::Sub;
use std::time::Duration;
use rand::prelude::*;
use std::rc::Rc;
use std::cell::{Ref, RefCell};
use std::cell::RefMut;
use std::collections::HashMap;
use nalgebra::{Isometry3, Matrix4, Perspective3, Vector2, Vector3, Vector4, Point3, Point2, Rotation3};
use std::fmt;

type EntityBehavior = fn(entity_id: usize, entites: &Entities, entities_diff: &mut EntitiesMut);
type Entities = HashMap<usize, Rc<dyn Entity>>;
type EntitiesMut = HashMap<usize, Box<dyn Entity>>;

type FVector4 = Vector4<f64>;
type FVector3 = Vector3<f64>;
type FVector2 = Vector2<f64>;
type FPoint3 = Point3<f64>;
type FPoint2 = Point2<f64>;

struct DrawContext {
    screen_center: FPoint2,
    eye: FPoint3,
    target: FPoint3,
    view: Isometry3<f64>,
    projection: Perspective3<f64>,
    canvas: Canvas<Window>
}
impl DrawContext {
    fn new(screen_center: FPoint2, camera_pos: FPoint3, fov: f64, canvas: Canvas<Window>) -> DrawContext {
        let target = camera_pos + Vector3::new(0.0_f64, 0.0_f64, 1.0_f64);

        DrawContext {
            screen_center,
            eye: camera_pos,
            target: target,
            view: Isometry3::look_at_rh(&camera_pos, &target, &Vector3::y()),
            projection: Perspective3::new(screen_center.x / screen_center.y, fov.to_radians(), 0.1, 10000.0),
            canvas
        }
    }
}
impl fmt::Debug for DrawContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DrawContext")
            .field("screen_center", &self.screen_center)
            .field("eye", &self.eye)
            .field("target", &self.target)
            .field("view", &self.view)
            .field("projection", &self.projection)
            .finish()
    }
}

trait Entity {
    fn pos(&mut self) -> &mut FPoint3;
    fn get_pos(&self) -> &FPoint3;
    fn vel(&mut self) -> &mut FVector3;
    fn get_vel(&self) -> &FVector3;
    fn acc(&mut self) -> &mut FVector3;
    fn get_acc(&self) -> &FVector3;
    fn col(&mut self) -> &mut Color;
    fn get_col(&self) -> &Color;
    fn rot(&mut self) -> &mut FVector3;
    fn get_rot(&self) -> &FVector3;
    fn rot_vel(&mut self) -> &mut FVector3;
    fn get_rot_vel(&self) -> &FVector3;
    fn draw(&self, ctx: &mut DrawContext);
    fn register_behavior(self: &mut Self, b: EntityBehavior);
    fn remove_behavior(self: &mut Self, b: EntityBehavior);
    fn behaviors(&self) -> &Vec<EntityBehavior>;
    fn clone(&self) -> Box<dyn Entity>;
}

struct Planet {
    pos: FPoint3,
    vel: FVector3,
    acc: FVector3,
    rot: FVector3,
    isom: Isometry3<f64>,
    rot_vel: FVector3,
    color: Color,
    behaviors: Vec<EntityBehavior>,
    rad: f64,
}

fn circle_height(x: f64, a: f64, b: f64, h: f64) -> f64{
    ((a*a) + a*x*2.0 + h*h - (x*x)).sqrt() + b
}

fn random(min: f64, max: f64) -> f64{
    min + rand::random::<f64>() * (max-min)
}

fn random_point_around(around: &FPoint3, range: f64) -> FPoint3{
    let offset = Vector3::new(random(-range, range), random(-range, range), random(-range, range));
    around + offset
}


fn random_color() -> Color {
    Color::RGB(
        random(15.0, 255.0) as u8,
        random(15.0, 255.0) as u8,
        random(15.0, 255.0) as u8
    )
}

fn draw_circle(center: &FPoint, radius: f64, rotation: &FVector3, color: &Color, ctx: &mut DrawContext) {
    let mut color = Color::RGB(color.r, color.g, color.b);
    let mut rx = rotation.y;
    for i in -radius as i32-5..radius as i32+5{
        let r = (rx.cos()+0.5).min(1.0).max(0.0);
        let c = Color::RGB(
            ((color.r as f64) *r) as u8, 
            ((color.g as f64) *r) as u8, 
            ((color.b as f64) *r) as u8, 
        );
        ctx.canvas.set_draw_color(c);
        let y = circle_height(i as f64, 1.0, 1.0, radius);
        ctx.canvas.draw_fline(
            center.offset(i as f32, -y as f32), 
            center.offset(i as f32, y as f32)
        );
        //rx += f64::consts::PI  / (radius*8.0);
    }
}

fn square_corners(center: &FPoint3, rotation: &FVector3, size: f64) -> Vec<FPoint3> {
    let mut corners = Vec::new();
    let rotation = Rotation3::from_euler_angles(rotation.x, rotation.y, rotation.z).to_homogeneous();
    
    for x in [-1.0, 1.0].iter() {
        for y in [-1.0, 1.0].iter() {
            let mut corner = Vector4::new(*x*size/2.0, *y*size/2.0, 0.0, 1.0);
            corner = rotation * corner;
            corners.push(FPoint3::new(
                center.x + corner.x,
                center.y + corner.y,
                center.z + corner.z
            ));
        }
    }
    vec![corners[0], corners[1], corners[3], corners[2]]
}

fn cube_faces(center: &FPoint3, rotation: &FVector3, size: f64) -> Vec<Vec<FPoint3>>{
    let mut faces = Vec::new();
    
    faces.insert(0, vec![
        FPoint3::new(-1.0, -1.0, 1.0),
        FPoint3::new(1.0, -1.0, 1.0),
        FPoint3::new(1.0, 1.0, 1.0),
        FPoint3::new(-1.0, 1.0, 1.0),
    ]);
    faces.insert(1, vec![
        FPoint3::new(-1.0, -1.0, -1.0),
        FPoint3::new(1.0, -1.0, -1.0),
        FPoint3::new(1.0, 1.0, -1.0),
        FPoint3::new(-1.0, 1.0, -1.0),
    ]);
    faces.insert(2,vec![
        FPoint3::new(-1.0, 1.0, -1.0),
        FPoint3::new(-1.0, 1.0, 1.0),
        FPoint3::new(1.0, 1.0, 1.0),
        FPoint3::new(1.0, 1.0, -1.0)
    ]);
    faces.insert(3,vec![
        FPoint3::new(-1.0, -1.0, -1.0),
        FPoint3::new(-1.0, -1.0, 1.0),
        FPoint3::new(1.0, -1.0, 1.0),
        FPoint3::new(1.0, -1.0, -1.0)
    ]);
    // faces.insert(4,vec![
    //     FPoint3::new(1.0, -1.0, -1.0),
    //     FPoint3::new(1.0, -1.0, 1.0),
    //     FPoint3::new(1.0, 1.0, 1.0),
    //     FPoint3::new(1.0, 1.0, -1.0)
    // ]);
    // faces.insert(5,vec![
    //     FPoint3::new(-1.0, -1.0, -1.0),
    //     FPoint3::new(-1.0, -1.0, 1.0),
    //     FPoint3::new(-1.0, 1.0, 1.0),
    //     FPoint3::new(-1.0, 1.0, -1.0)
    // ]);

    for f in &mut faces {
        for p in f {
            *p = Isometry3::new(center.coords, *rotation) * *p;
        }
    }

    for f in &mut faces {
        for p in f {
           // *p = (*p * size) + center.coords;
        }
    }

    faces
}

fn draw_polygon(corners: &Vec<FPoint3>, size: f64, rotation: &FVector3, color: &Color, ctx: &mut DrawContext) {
    let corners = corners.iter().map(|p| point_on_screen(p, ctx)).collect::<Vec<FPoint3>>();
    let mut color = Color::RGB(color.r, color.g, color.b);
    println!("{:?}",corners.len());
    //println!("{:?}", corners);
    for i in 0..corners.len() {
        let a = corners[i % corners.len()];
        let b = corners[(i+1) % corners.len()];

        if !sdl_on_screen(&a, ctx) && !sdl_on_screen(&b, ctx) {
            continue;
        }

        let a = sdl_point(&a);
        let b = sdl_point(&b);

        ctx.canvas.set_draw_color(color);
        ctx.canvas.draw_fline(a, b);
        draw_circle(&a, 1.0, rotation, &color, ctx);
        println!("draw from {a:?} to {b:?}");
    }
}

fn point_on_screen(p: &Point3<f64>, ctx: &DrawContext) -> FPoint3 {
    FPoint3::new(
        (p.x + ctx.eye.x + ctx.screen_center.x) as f64,
        (p.y + ctx.eye.y + ctx.screen_center.y) as f64,
        0.0
    )
}

fn sdl_point(p: &FPoint3) -> FPoint {
    FPoint::new(
        (p.x) as f32,
        (p.y) as f32
    )
}

fn sdl_on_screen(p: &FPoint3, ctx: &DrawContext) -> bool {
    p.z >= 0.0 && (p.x > 0.0 && p.x < ctx.screen_center.x * 2.0 && p.y > 0.0 && p.y < ctx.screen_center.y * 2.0)
}

fn project_point(p: &Point3<f64>, ctx: &DrawContext) -> Point3<f64> { 
    ctx.projection.project_point(p)
}

impl Entity for Planet {
    fn pos(&mut self) -> &mut FPoint3 {
        &mut self.pos
    }
    fn get_pos(&self) -> &FPoint3 {
        &self.pos
    }
    fn vel(&mut self) -> &mut FVector3 {
        &mut self.vel
    }
    fn get_vel(&self) -> &FVector3 {
        &self.vel
    }
    fn acc(&mut self) -> &mut FVector3 {
        &mut self.acc
    }
    fn get_acc(&self) -> &FVector3 {
        &self.acc
    }
    fn rot(&mut self) -> &mut FVector3 {
        &mut self.rot
    }
    fn get_rot(&self) -> &FVector3 {
        &self.rot
    }
    fn rot_vel(&mut self) -> &mut FVector3 {
        &mut self.rot_vel
    }
    fn get_rot_vel(&self) -> &FVector3 {
        &self.rot_vel
    }
    fn col(&mut self) -> &mut Color {
        &mut self.color
    }
    fn get_col(&self) -> &Color {
        &self.color
    }
    fn draw(&self, ctx: &mut DrawContext) {
        if self.pos.z < 0.0 {
            return;
        }
        ctx.canvas.set_draw_color(self.color);
        // draw 3d coordinates object on 2d screen

        //let size = self.rad * self.pos.z / 100.0;
        //let corners = square_corners(&self.pos, &self.rot, self.rad)
        let faces = cube_faces(&self.pos, &self.rot, self.rad);
        for face in faces {
            let corners = face.iter().map(|p| project_point(p, ctx)).collect::<Vec<FPoint3>>();
            draw_polygon(&corners, self.rad, &self.rot, &self.color, ctx);
        }
        
        
    }
    fn register_behavior(&mut self, b: EntityBehavior){
        self.behaviors.push(b);
    }
    fn remove_behavior(self: &mut Self, b: EntityBehavior) {
        self.behaviors.retain(|&x| x != b);
    }
    fn behaviors(&self) -> &Vec<EntityBehavior> {
        &self.behaviors
    }
    fn clone(&self) -> Box<dyn Entity> {
        Box::new(Planet{
            pos: self.pos.clone(),
            vel: self.vel.clone(),
            acc: self.acc.clone(),
            rot: self.rot.clone(),
            rot_vel: self.rot_vel.clone(),
            color: self.color.clone(),
            behaviors: self.behaviors.clone(),
            rad: self.rad,
            isom: self.isom.clone()
        })
    }
}

fn behavior_movement(id: usize, entites: &Entities, entities_diff: &mut EntitiesMut) {

    let mut diff: Box<dyn Entity>;
    if let Some(existing) = entities_diff.get_mut(&id) {
        diff = existing.clone();
    } else {
        diff = (*(entites.get(&id).unwrap().clone())).clone().clone();
    }

    let a = *diff.acc();
    *diff.vel() += a;

    let v = *diff.vel();
    *diff.pos() += v;

    let r = *diff.rot_vel();
    *diff.rot() += r;



    diff.vel().scale(0.99);
    diff.acc().scale(0.99);
    
    entities_diff.insert(id, diff);   
}

fn behavior_gravity(id: usize, entites: &Entities, entities_diff: &mut EntitiesMut) {
    let mut diff: Box<dyn Entity>;
    if let Some(existing) = entities_diff.get_mut(&id) {
        diff = existing.clone();
    } else {
        diff = (*(entites.get(&id).unwrap().clone())).clone().clone();
    }

    for (k, v) in entites {
        if k == &id {
            continue;
        }
        let e = v;
        let dx = e.get_pos().x - diff.get_pos().x;
        let dy = e.get_pos().y - diff.get_pos().y;
        let dz = e.get_pos().z - diff.get_pos().z;
        let d = (dx*dx + dy*dy + dz*dz).sqrt();
        let f = 0.00002 / d;
        let fx = f * dx;
        let fy = f * dy;
        let fz = f * dz;
        diff.acc().x += fx;
        diff.acc().y += fy;
        diff.acc().z += fz;
    }

    entities_diff.insert(id, diff);
}

fn tick_entity(
    i: usize,
    w: &Entities,
    w_mut: &mut EntitiesMut,
) {
    let e = w.get(&i).unwrap();
    let b = e.behaviors().to_vec();
    b.iter().for_each(|b| b(i, w, w_mut));
}

fn tick(w: &mut Entities) {
    let mut w_diff : EntitiesMut = HashMap::new();

    for i in 0..w.len() {
        tick_entity(i, &w, &mut w_diff);
    }

    for (k, v) in w_diff {
        w.insert(k, Rc::from(v));
    }
}

pub fn main() -> Result<(), String> {
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video();
    let screen_center = FPoint2::new(400.0, 320.0);

    let window = video_subsystem?
    .window("rust sdl2 demo: Video", screen_center.x as u32 * 2, screen_center.y as u32 * 2)
    .position_centered()
    .opengl()
    .build()
    .map_err(|e| e.to_string()
    )?;

    let mut canvas = window
    .into_canvas()
    .build()
    .map_err(|e| e.to_string()
    )?;

    let mut draw_ctx = DrawContext::new(
        screen_center,
        FPoint3::new(0.0, 0.0, 0.0),
        0.5,
        canvas
    );

    let bg_color = Color::RGB(0, 25, 36);
    draw_ctx.canvas.set_draw_color(bg_color);
    draw_ctx.canvas.clear();

    let fps = 30.0;
    let mut event_loop = sdl_context.event_pump()?;
    let center = FPoint3::new(400.0, 320.0, 100.0);

    let mut entities: Entities = HashMap::new();
    let mut entity_counter: usize = 0;

    let simulation_start = std::time::Instant::now();

    for i in 0..100 {
        let rot = random_point_around(&FPoint3::new(0.0, 0.0, 0.0), 1.0).coords;
        let pos =  random_point_around(&FPoint3::new(0.0, 0.0, 10.0), 0.0);
        let mut planet = Planet{
            pos,
            vel: random_point_around(&FPoint3::new(0.0, 0.0, 0.0), 0.1).coords,
            acc: FVector3::new(0.0, 0.0, 0.0),
            rot,
            rot_vel: random_point_around(&FPoint3::new(0.0, 0.1, 0.0), 0.0).coords,
            color: random_color(),
            behaviors: vec![behavior_movement],
            rad: random(1.0, 10.0),
            isom: Isometry3::new(pos.coords, rot)
        };
        entities.insert(entity_counter, Rc::new(planet));
        entity_counter += 1;
    }

    let mut tick_times: Vec<Duration> = Vec::new();
    let mut tick_lengths: Vec<Duration> = Vec::new();
    'running: loop {
        let tick_time_start = std::time::Instant::now();

        for event in event_loop.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                _ => {}
            }
        }

        tick(&mut entities);

        let mut sorted: Vec<(&usize, &Rc<dyn Entity>)> = entities.iter().collect();
        sorted.sort_by(|a, b| {
            let a = a.1.get_pos();
            let b = b.1.get_pos();
            if a.z < b.z {
                return Ordering::Greater
            } else if a.z < b.z {
                return Ordering::Less
            }
            Ordering::Equal
        });

        draw_ctx.canvas.set_draw_color(bg_color);
        draw_ctx.canvas.clear();

        for e in &entities
        {
            e.1.draw(&mut draw_ctx);
        }

        draw_ctx.canvas.present();

        if tick_times.len() > 100 {
            tick_times.remove(0);
        }
        if tick_lengths.len() > 100 {
            tick_lengths.remove(0);
        }

        let tick_time_end = std::time::Instant::now();
        tick_times.push(tick_time_end - simulation_start);
        tick_lengths.push(tick_time_end - tick_time_start);


        // println!("tick time: {:?}, calculated fps: {:?}, real fps: {:?}, target fps: {:?}", 
        //     tick_lengths.iter().sum::<Duration>() / tick_lengths.len() as u32,
        //     1e9 * tick_lengths.len() as f32 / tick_lengths.iter().sum::<Duration>().as_nanos() as f32,
        //     1e9 * tick_times.len() as f32 / (*tick_times.iter().last().unwrap() - *tick_times.iter().nth(0).unwrap()).as_nanos() as f32,
        //     fps
        // );
        
        let sleep_duration = Duration::checked_sub(
            Duration::new(0, (1e9/fps) as u32),
            std::time::Instant::now() - tick_time_start
        );
            
        if let Some(sleep_duration) = sleep_duration {
            std::thread::sleep(sleep_duration);
        }
    }

    Ok(())
}