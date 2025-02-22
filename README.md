# GO-Physlicks (WIP)

## Features

GO-Physlicks is a lightweight 2D physics engine written in Go, designed for game development. Here are its key features:

***Note: This is still a work in progress.***

## Core Physics Components

- **Rigid Body Dynamics**: Handles movement, rotation, and collision responses for game objects
- **Vector Mathematics**: Provides essential vector operations for physics calculations
- **Collision Detection**: Implements efficient AABB and circle collision detection algorithms

## Key Benefits

1. **Performance**
   - Optimized for 2D games
   - Minimal memory allocation
   - Efficient broad-phase collision detection

2. **Simplicity**
   - Clean, readable API
   - Easy integration with existing Go projects
   - Minimal dependencies

3. **Flexibility**
   - Customizable physics parameters
   - Support for different collision shapes
   - Extensible system for custom physics behaviors

## Usage Examples

Here's a basic example of how to use GO-Physlicks to create a simple physics simulation:

```go
package main

import (
    physics "github.com/edwinsyarief/go-physlicks"
)

func main() {
    // Initialize physics system with screen bounds and quadtree parameters
    bounds := physics.AABB{Min: physics.Point{X: 0, Y: 0}, Max: physics.Point{X: 800, Y: 600}}
    physicsSystem := physics.NewPhysicsSystem(bounds, 4, 8)

    // Create and add a circle
    circle := physics.NewCircleShape(physics.Point{X: 100, Y: 100}, 20)
    physicsSystem.AddShape(circle)

    // Create and add a rectangle
    rect := physics.NewRectShape(physics.Point{X: 400, Y: 300}, 40, 40, 0)
    physicsSystem.AddShape(rect)

    // In your game loop:
    // 1. Detect collisions
    collidingPairs := physicsSystem.DetectCollisions()

    // 2. Handle collisions and update physics
    for _, pair := range collidingPairs {
        // Implement collision response
        // See examples/basic/main.go for detailed collision handling
    }
}
```

This example demonstrates:
- Initializing the physics system
- Creating and adding basic shapes (circle and rectangle)
- Detecting collisions between shapes

For a more complete example with full collision handling, velocity updates, and rendering, check out `examples/basic/main.go`.