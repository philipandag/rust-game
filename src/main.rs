use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::{FRect, FPoint};
use sdl2::render::{Canvas};
use sdl2::video::Window;
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

#[derive(Debug, Clone)]
struct FPoint3{
    x: f32,
    y: f32,
    z: f32
}

type EntityBehavior = fn(entity_id: usize, entites: &Entities, entities_diff: &mut EntitiesMut);
type Entities = HashMap<usize, Rc<dyn Entity>>;
type EntitiesMut = HashMap<usize, Box<dyn Entity>>;

struct DrawContext {
    screen_center: FPoint3,
    camera_pos: FPoint3,
    fov: f32,
}

trait Entity {
    fn pos(&mut self) -> &mut FPoint3;
    fn get_pos(&self) -> &FPoint3;
    fn vel(&mut self) -> &mut FPoint3;
    fn get_vel(&self) -> &FPoint3;
    fn acc(&mut self) -> &mut FPoint3;
    fn get_acc(&self) -> &FPoint3;
    fn col(&mut self) -> &mut Color;
    fn get_col(&self) -> &Color;
    fn draw(&self, c: &mut Canvas<Window>, ctx: &DrawContext);
    fn register_behavior(self: &mut Self, b: EntityBehavior);
    fn remove_behavior(self: &mut Self, b: EntityBehavior);
    fn behaviors(&self) -> &Vec<EntityBehavior>;
    fn clone(&self) -> Box<dyn Entity>;
}

struct Planet {
    pos: FPoint3,
    vel: FPoint3,
    acc: FPoint3,
    color: Color,
    behaviors: Vec<EntityBehavior>,
    rad: f32,
}

fn circle_height(x: &f32, a: &f32, b: &f32, h: &f32)  -> f32{
    ((a*a) + a*x*2.0 + h*h - (x*x)).sqrt() + b
}

fn random(min: f32, max: f32) -> f32{
    min + rand::random::<f32>() * (max-min)
}

fn random_point_around(around: &FPoint3, range: f32) -> FPoint3{
    FPoint3 {
        x: around.x + random(-range, range),
        y: around.y + random(-range, range),
        z: around.z + random(-range, range),
    }
}

fn random_color() -> Color {
    Color::RGB(
        random(15.0, 255.0) as u8,
        random(15.0, 255.0) as u8,
        random(15.0, 255.0) as u8
    )
}

fn draw_circle(center: &FPoint, radius: &f32, color: &Color, canvas: &mut Canvas<Window>) {
    let mut color = Color::RGB(color.r, color.g, color.b);
    for i in -*radius as i32..*radius as i32 +5{
        color = Color::RGB(
            ((color.r as f32) *0.995) as u8, 
            ((color.g as f32) *0.995) as u8, 
            ((color.b as f32) *0.995) as u8, 
        );
        canvas.set_draw_color(color);
        let y = circle_height(&(i as f32), &1.0, &1.0, &radius);
        canvas.draw_fline(
            center.offset(i as f32, -y), 
            center.offset(i as f32, y)
        );
    }
}

impl Entity for Planet {
    fn pos(&mut self) -> &mut FPoint3 {
        &mut self.pos
    }
    fn get_pos(&self) -> &FPoint3 {
        &self.pos
    }
    fn vel(&mut self) -> &mut FPoint3 {
        &mut self.vel
    }
    fn get_vel(&self) -> &FPoint3 {
        &self.vel
    }
    fn acc(&mut self) -> &mut FPoint3 {
        &mut self.acc
    }
    fn get_acc(&self) -> &FPoint3 {
        &self.acc
    }
    fn col(&mut self) -> &mut Color {
        &mut self.color
    }
    fn get_col(&self) -> &Color {
        &self.color
    }
    fn draw(&self, c: &mut Canvas<Window>, ctx: &DrawContext) {
        c.set_draw_color(self.color);
        // draw 3d coordinates object on 2d screen

        let pitch = ((self.pos.x - ctx.camera_pos.x) / (self.pos.y - ctx.camera_pos.y)).atan();
        let yaw = ((self.pos.z - ctx.camera_pos.z) / (self.pos.y - ctx.camera_pos.y)).atan();
        
        let pos2d = FPoint::new(
            ctx.screen_center.x + (pitch * (ctx.screen_center.x * 2.0 / ctx.fov)),
            ctx.screen_center.y + (yaw * (ctx.screen_center.y * 2.0 / ctx.fov))
        );

        let size = self.rad * self.pos.z / 100.0;
        

        draw_circle(&pos2d, &size, &self.color, c);
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
            color: self.color.clone(),
            behaviors: self.behaviors.clone(),
            rad: self.rad
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

    diff.vel().x += diff.acc().x;
    diff.vel().y += diff.acc().y;
    diff.vel().z += diff.acc().z;

    diff.pos().x += diff.vel().x;
    diff.pos().y += diff.vel().y;
    diff.pos().z += diff.vel().z;

    diff.vel().x *= 0.99;
    diff.vel().y *= 0.99;
    diff.vel().z *= 0.99;
    diff.acc().x *= 0.99;
    diff.acc().y *= 0.99;
    diff.acc().z *= 0.99;


    
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
    let mut draw_context = DrawContext {
        screen_center: FPoint3{x: 400.0, y: 320.0, z: 0.0},
        camera_pos: FPoint3{x: 0.0, y: 0.0, z: 0.0},
        fov: 10.0,
    };

    let window = video_subsystem?
        .window("rust sdl2 demo: Video", draw_context.screen_center.x as u32 * 2, draw_context.screen_center.y as u32 * 2)
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


    let bg_color = Color::RGB(222, 195, 199);
    canvas.set_draw_color(bg_color);
    canvas.clear();
    canvas.present();

    let fps = 1000.0;
    let mut event_loop = sdl_context.event_pump()?;
    let center = FPoint3{x: 400.0, y: 320.0, z: 100.0};

    let mut entities: Entities = HashMap::new();
    let mut entity_counter: usize = 0;

    let simulation_start = std::time::Instant::now();

    for i in 0..10 {
        let mut planet = Planet{
            pos: random_point_around(&FPoint3{x:0.0, y:0.0, z:90.0}, 10.0),
            vel: random_point_around(&FPoint3{x:0.0, y:0.0, z:0.0}, 50.0),
            acc: FPoint3{x:0.0, y:0.0, z:0.0},
            color: random_color(),
            behaviors: vec![behavior_movement],
            rad: random(1.0, 2.0)
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

        canvas.set_draw_color(bg_color);
        canvas.clear();

        for e in &entities
        {
            e.1.draw(&mut canvas, &draw_context);
        }

        canvas.present();

        if tick_times.len() > 100 {
            tick_times.remove(0);
        }
        if tick_lengths.len() > 100 {
            tick_lengths.remove(0);
        }

        let tick_time_end = std::time::Instant::now();
        tick_times.push(tick_time_end - simulation_start);
        tick_lengths.push(tick_time_end - tick_time_start);


        println!("tick time: {:?}, calculated fps: {:?}, real fps: {:?}, target fps: {:?}", 
            tick_lengths.iter().sum::<Duration>() / tick_lengths.len() as u32,
            1e9 * tick_lengths.len() as f32 / tick_lengths.iter().sum::<Duration>().as_nanos() as f32,
            1e9 * tick_times.len() as f32 / (*tick_times.iter().last().unwrap() - *tick_times.iter().nth(0).unwrap()).as_nanos() as f32,
            fps
        );
        
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