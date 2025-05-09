//! Spawn the main level.

use bevy::{prelude::*, scene::SceneInstanceReady};

use crate::{
    props::*, screens::Screen, third_party::bevy_trenchbroom::GetTrenchbroomModelPath as _,
};

pub(super) fn plugin(app: &mut App) {
    app.init_resource::<LevelAssets>();
    app.register_type::<Level>();
}

/// A [`Command`] to spawn the level.
/// Functions that accept only `&mut World` as their parameter implement [`Command`].
/// We use this style when a command requires no configuration.
pub(crate) fn spawn_level(world: &mut World) {
    let assets = world.resource::<LevelAssets>();
    world
        .spawn((
            Name::new("Level"),
            SceneRoot(assets.level.clone()),
            StateScoped(Screen::Gameplay),
            Level,
        ))
        .observe(advance_to_gameplay_screen);
}

#[derive(Debug, Component, Reflect)]
#[reflect(Component)]
pub(crate) struct Level;

fn advance_to_gameplay_screen(
    _trigger: Trigger<SceneInstanceReady>,
    mut next_screen: ResMut<NextState<Screen>>,
) {
    next_screen.set(Screen::Gameplay);
}

#[derive(Resource, Asset, Clone, Reflect)]
#[reflect(Resource)]
struct LevelAssets {
    #[dependency]
    pub(crate) level: Handle<Scene>,
    #[dependency]
    pub(crate) props: Vec<Handle<Scene>>,
}

impl FromWorld for LevelAssets {
    fn from_world(world: &mut World) -> Self {
        let assets = world.resource::<AssetServer>();
        Self {
            //  Run ./scripts/compile_maps.sh and change .map to .bsp when we're done prototyping and want some extra performance
            level: assets.load("maps/foxtrot/foxtrot.map#Scene"),
            // We preload all props used in the level here. The template is setup such that we get a helpful warning if we miss one.
            props: vec![
                assets.load(Book::scene_path()),
                assets.load(Candle::scene_path()),
                assets.load(CandleUnlit::scene_path()),
                assets.load(Mug::scene_path()),
                assets.load(Plate::scene_path()),
                assets.load(Drawers::scene_path()),
            ],
        }
    }
}
