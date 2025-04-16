// vim: tw=80
//! A generic struct with a generic method on a different parameter
#![deny(warnings)]

use mockall::*;
use std::sync::Mutex;

mock! {
    Foo<T: 'static + std::fmt::Debug> {
        fn foo<Q: 'static + std::fmt::Debug>(t: T, q: Q) -> u64;
    }
}

static FOO_MTX: Mutex<()> = Mutex::new(());

// Checkpointing the mock object should not checkpoint static methods too
#[test]
fn checkpoint() {
    let _m = FOO_MTX.lock();

    let mut mock = MockFoo::<u32>::new();
    let ctx = MockFoo::<u32>::foo_context();
    ctx.expect::<i16>()
        .returning(|_, _| 0)
        .times(1..3);
    mock.checkpoint();
    MockFoo::foo(42u32, 69i16);
}

// It should also be possible to checkpoint just the context object
#[test]
#[should_panic(expected =
    "MockFoo::foo: Expectation(<anything>) called 0 time(s) which is fewer than expected 1")]
fn ctx_checkpoint() {
    let _m = FOO_MTX.lock();
    let ctx = MockFoo::<u32>::foo_context();
    ctx.expect::<i16>()
        .returning(|_, _| 0)
        .times(1..3);
    ctx.checkpoint();
    panic!("Shouldn't get here!");
}

// Expectations should be cleared when a context object drops
#[test]
#[should_panic(expected = "MockFoo::foo(42, 69): No matching expectation found")]
fn ctx_hygiene() {
    let _m = FOO_MTX.lock();
    {
        let ctx0 = MockFoo::<u32>::foo_context();
        ctx0.expect::<i16>()
            .returning(|_, _| 0);
    }
    MockFoo::foo(42, 69);
}

#[cfg_attr(not(feature = "nightly"), ignore)]
#[cfg_attr(not(feature = "nightly"), allow(unused_must_use))]
#[test]
fn return_default() {
    let _m = FOO_MTX.lock();

    let ctx = MockFoo::<u32>::foo_context();
    ctx.expect::<i16>();
    MockFoo::foo(5u32, 6i16);
}

#[test]
fn returning() {
    let _m = FOO_MTX.lock();

    let ctx = MockFoo::<u32>::foo_context();
    ctx.expect::<i16>()
        .returning(|_, _| 0);
    MockFoo::foo(41u32, 42i16);
}

#[test]
fn two_matches() {
    let _m = FOO_MTX.lock();

    let ctx = MockFoo::<u32>::foo_context();
    ctx.expect::<i16>()
        .with(predicate::eq(42u32), predicate::eq(0i16))
        .return_const(99u64);
    ctx.expect::<i16>()
        .with(predicate::eq(69u32), predicate::eq(0i16))
        .return_const(101u64);
    assert_eq!(101, MockFoo::foo(69u32, 0i16));
    assert_eq!(99, MockFoo::foo(42u32, 0i16));
}

#[test]
fn with() {
    let _m = FOO_MTX.lock();

    let ctx = MockFoo::<u32>::foo_context();
    ctx.expect::<i16>()
        .with(predicate::eq(42u32), predicate::eq(99i16))
        .returning(|_, _| 0);
    MockFoo::foo(42u32, 99i16);
}
