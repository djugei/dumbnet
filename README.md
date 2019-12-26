
# DumbNet

<!-- cargo-sync-readme start -->

A Neural Network library that does not make use of allocations or the standard library at all.
It does all its work on the stack.

This has some advantages:

## Embedded
Embedded devices without any operating system are now able to run at least simple neural
networks.

## Compile-Time checks
Since the whole network layout needs to be known at compile time the dimensions of inputs and
outputs are checked.

## Optimization
The whole network being known to the compiler might enable some optimizations. That said the
library is currently not very well optimized.

<!-- cargo-sync-readme end -->

## Plans
- [ ] a convolutional layer would be nice
- [ ] think about the library design, specifically Layer might be too coarse of a trait, sub-layers may be useful.
  - [ ] unify SoftMax and other layers
- [ ] better optimization
- [ ] ! move to less horrible generics

## 1.0 Statement
This crate will be 1.0 if it has the tools to detect handwriting and is kinda easy to use.

# Contributing
Please symlink the hooks to your local .git/hooks/ directory to run some automatic checks before committing.

    ln -s ../../hooks/pre-commit .git/hooks/

Please install rustfmt and cargo-sync-readme so these checks can be run.

    rustup component add rustfmt
    cargo install cargo-sync-readme

Please execute `cargo-sync-readme` when you change the top-level-documentation.
Please run `cargo fmt` whenever you change code. If possible configure your editor to do so for you.
