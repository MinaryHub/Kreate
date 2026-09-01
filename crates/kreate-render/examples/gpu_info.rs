//! Prints every GPU adapter Kreate can see, and initialises a device.
//!
//! Run with `cargo run -p kreate-render --example gpu_info`. Useful as a first
//! diagnostic when a user reports poor performance: on a hybrid laptop the
//! answer is very often that the integrated GPU was picked.

fn main() {
    let opts = kreate_render::GpuOptions::default();
    let instance = kreate_render::Gpu::create_instance(&opts);

    println!("backends requested: {:?}\n", opts.backends);
    let adapters = pollster::block_on(kreate_render::Gpu::enumerate(&instance, &opts));
    if adapters.is_empty() {
        println!("no adapters found");
        return;
    }
    println!("adapters visible ({}):", adapters.len());
    for a in &adapters {
        println!(
            "  {}{}\n      driver: {}",
            a.describe(),
            if a.is_discrete { "  <- discrete" } else { "" },
            a.driver
        );
    }

    println!("\ninitialising a device with the default preference...");
    match pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)) {
        Ok(gpu) => {
            println!("  selected: {}", gpu.info.describe());
            if !gpu.info.is_discrete && adapters.iter().any(|a| a.is_discrete) {
                println!("  WARNING: an integrated GPU was selected while a discrete one exists.");
                println!("           This is the hybrid-GPU trap; check the vendor hint exports.");
            }
            let limits = gpu.device.limits();
            println!(
                "  max buffer size: {} MiB, max texture 2D: {}",
                limits.max_buffer_size / (1024 * 1024),
                limits.max_texture_dimension_2d
            );
        }
        Err(e) => println!("  FAILED: {e}"),
    }
}
