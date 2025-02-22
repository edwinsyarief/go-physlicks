package main

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"time"

	physics "github.com/edwinsyarief/go-physlicks"
	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
)

// Game represents the main game structure.
type Game struct {
	physics       *physics.PhysicsSystem
	velocities    map[int]physics.Point // Linear velocity
	angVelocities map[int]float64       // Angular velocity (radians per frame)
	screenWidth   int
	screenHeight  int
	gravity       physics.Point
}

// Update updates the game state.
func (g *Game) Update() error {
	// Apply gravity to velocities
	for _, shape := range g.physics.GetShapes() {
		vel := g.velocities[shape.GetID()]
		g.velocities[shape.GetID()] = vel.Add(g.gravity)
	}

	// Update positions and rotations
	for _, shape := range g.physics.GetShapes() {
		vel := g.velocities[shape.GetID()]
		angVel := g.angVelocities[shape.GetID()]
		shape.Move(vel)
		shape.SetRotation(shape.GetRotation() + angVel)

		// Clamp to screen boundaries
		aabb := shape.GetAABB()
		if aabb.Min.X < 0 {
			shape.Move(physics.Point{X: -aabb.Min.X, Y: 0})
			g.velocities[shape.GetID()] = physics.Point{X: -vel.X * shape.GetElasticity(), Y: vel.Y}
			g.angVelocities[shape.GetID()] = -angVel * shape.GetElasticity()
		}
		if aabb.Max.X > float64(g.screenWidth) {
			shape.Move(physics.Point{X: float64(g.screenWidth) - aabb.Max.X, Y: 0})
			g.velocities[shape.GetID()] = physics.Point{X: -vel.X * shape.GetElasticity(), Y: vel.Y}
			g.angVelocities[shape.GetID()] = -angVel * shape.GetElasticity()
		}
		if aabb.Min.Y < 0 {
			shape.Move(physics.Point{X: 0, Y: -aabb.Min.Y})
			g.velocities[shape.GetID()] = physics.Point{X: vel.X, Y: -vel.Y * shape.GetElasticity()}
			g.angVelocities[shape.GetID()] = -angVel * shape.GetElasticity()
		}
		if aabb.Max.Y > float64(g.screenHeight) {
			shape.Move(physics.Point{X: 0, Y: float64(g.screenHeight) - aabb.Max.Y})
			g.velocities[shape.GetID()] = physics.Point{X: vel.X, Y: -vel.Y * shape.GetElasticity()}
			g.angVelocities[shape.GetID()] = -angVel * shape.GetElasticity()
		}
	}

	// Handle collisions
	collidingPairs := g.physics.DetectCollisions()
	for _, pair := range collidingPairs {
		resolveCollision(pair[0], pair[1], g.velocities, g.angVelocities)
	}

	return nil
}

// Draw renders the game.
func (g *Game) Draw(screen *ebiten.Image) {
	shapes := g.physics.GetShapes()
	for _, shape := range shapes {
		switch shape.GetType() {
		case physics.CircleType:
			circle := shape.(*physics.CircleShape)
			pos := circle.GetPosition()
			radius := circle.Radius
			drawCircle(screen, pos.X, pos.Y, radius, shape.GetColor())
		case physics.PolygonType:
			poly := shape.(*physics.PolygonShape)
			verts := poly.GetWorldVertices()
			drawPolygon(screen, verts, shape.GetColor())
		}
	}

	ebitenutil.DebugPrint(screen, fmt.Sprintf("TPS: %f\nFPS: %f\nTotal Objects: %d", ebiten.ActualTPS(), ebiten.ActualFPS(), len(shapes)))
}

// drawCircle draws a circle using vector graphics.
func drawCircle(screen *ebiten.Image, x, y, radius float64, clr color.Color) {
	const numSegments = 32
	for i := 0; i < numSegments; i++ {
		angle1 := float64(i) * 2 * math.Pi / numSegments
		angle2 := float64(i+1) * 2 * math.Pi / numSegments
		x1 := x + radius*math.Cos(angle1)
		y1 := y + radius*math.Sin(angle1)
		x2 := x + radius*math.Cos(angle2)
		y2 := y + radius*math.Sin(angle2)
		ebitenutil.DrawLine(screen, x1, y1, x2, y2, clr)
	}
}

// drawPolygon draws a polygon using its vertices.
func drawPolygon(screen *ebiten.Image, vertices []physics.Point, clr color.Color) {
	if len(vertices) < 2 {
		return
	}
	for i := 0; i < len(vertices); i++ {
		v1 := vertices[i]
		v2 := vertices[(i+1)%len(vertices)]
		ebitenutil.DrawLine(screen, v1.X, v1.Y, v2.X, v2.Y, clr)
	}
}

// Layout sets the screen dimensions.
func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	return g.screenWidth, g.screenHeight
}

// resolveCollision resolves the collision with impulse-based physics including rotation.
func resolveCollision(shape1, shape2 physics.Shape, velocities map[int]physics.Point, angVelocities map[int]float64) {
	// Get velocities and properties
	v1 := velocities[shape1.GetID()]
	v2 := velocities[shape2.GetID()]
	w1 := angVelocities[shape1.GetID()]
	w2 := angVelocities[shape2.GetID()]
	m1 := shape1.GetMass()
	m2 := shape2.GetMass()
	i1 := shape1.GetMomentOfInertia()
	i2 := shape2.GetMomentOfInertia()
	e := (shape1.GetElasticity() + shape2.GetElasticity()) / 2

	// Approximate contact point as midpoint (for simplicity; could refine with SAT)
	contact := shape1.GetPosition().Add(shape2.GetPosition()).Scale(0.5)

	// Vectors from centers to contact point
	r1 := contact.Sub(shape1.GetPosition())
	r2 := contact.Sub(shape2.GetPosition())

	// Collision normal using accurate computation
	normal := physics.GetCollisionNormal(shape1, shape2)

	// Relative velocity at contact point, including rotational effects
	v1AtContact := v1.Add(r1.Perpendicular().Scale(w1))
	v2AtContact := v2.Add(r2.Perpendicular().Scale(w2))
	rv := v2AtContact.Sub(v1AtContact)
	vn := physics.Dot(rv, normal)

	// If moving apart, no impulse needed
	if vn > 0 {
		return
	}

	// Impulse calculation
	// j = -(1 + e) * vn / (1/m1 + 1/m2 + (r1 x n)^2 / I1 + (r2 x n)^2 / I2)
	r1CrossN := r1.X*normal.Y - r1.Y*normal.X
	r2CrossN := r2.X*normal.Y - r2.Y*normal.X
	denominator := (1/m1 + 1/m2) + (r1CrossN*r1CrossN)/i1 + (r2CrossN*r2CrossN)/i2
	if denominator == 0 {
		return
	}
	j := -(1 + e) * vn / denominator

	// Apply impulse
	impulse := normal.Scale(j)
	if m1 > 0 {
		v1 = v1.Sub(impulse.Scale(1 / m1))
	}
	if m2 > 0 {
		v2 = v2.Add(impulse.Scale(1 / m2))
	}
	if i1 > 0 {
		w1 -= r1CrossN * j / i1
	}
	if i2 > 0 {
		w2 += r2CrossN * j / i2
	}

	// Update velocities
	velocities[shape1.GetID()] = v1
	velocities[shape2.GetID()] = v2
	angVelocities[shape1.GetID()] = w1
	angVelocities[shape2.GetID()] = w2

	// Position correction
	physics.OverlapAdjustment(shape1, shape2, normal)
}

func main() {
	const (
		screenWidth      = 1280
		screenHeight     = 720
		numCircles       = 120
		numRects         = 120
		numPolygons      = 120
		quadtreeCapacity = 4
		quadtreeMaxDepth = 8
	)

	rand.Seed(time.Now().UnixNano())
	bounds := physics.AABB{Min: physics.Point{X: 0, Y: 0}, Max: physics.Point{X: screenWidth, Y: screenHeight}}
	gravity := physics.Point{X: 0, Y: 0} // Downward gravity
	physicsSystem := physics.NewPhysicsSystem(bounds, quadtreeCapacity, quadtreeMaxDepth)
	velocities := make(map[int]physics.Point)
	angVelocities := make(map[int]float64)

	// Add circles
	for i := 0; i < numCircles; i++ {
		pos := physics.Point{X: rand.Float64() * screenWidth, Y: rand.Float64() * screenHeight}
		radius := 5 + rand.Float64()*15
		circle := physics.NewCircleShape(pos, radius)
		physicsSystem.AddShape(circle)
		velocities[circle.GetID()] = physics.Point{X: rand.Float64()*4 - 2, Y: rand.Float64()*4 - 2}
		angVelocities[circle.GetID()] = (rand.Float64()*2 - 1) * 0.1
	}

	// Add rectangles
	for i := 0; i < numRects; i++ {
		pos := physics.Point{X: rand.Float64() * screenWidth, Y: rand.Float64() * screenHeight}
		width := 10 + rand.Float64()*30
		height := 10 + rand.Float64()*30
		rotation := rand.Float64() * 2 * math.Pi
		rect := physics.NewRectShape(pos, width, height, rotation)
		physicsSystem.AddShape(rect)
		velocities[rect.GetID()] = physics.Point{X: rand.Float64()*4 - 2, Y: rand.Float64()*4 - 2}
		angVelocities[rect.GetID()] = (rand.Float64()*2 - 1) * 0.1
	}

	// Add regular pentagons and triangles
	for i := 0; i < numPolygons; i++ {
		pos := physics.Point{X: rand.Float64() * screenWidth, Y: rand.Float64() * screenHeight}
		radius := 10 + rand.Float64()*20
		rotation := rand.Float64() * 2 * math.Pi

		// Create pentagon or triangle vertices
		var numVertices = 5
		if i%2 == 0 {
			numVertices = 3
		}
		vertices := make([]physics.Point, numVertices)
		for j := 0; j < numVertices; j++ {
			angle := float64(j) * 2 * math.Pi / float64(numVertices)
			vertices[j] = physics.Point{X: radius * math.Cos(angle), Y: radius * math.Sin(angle)}
		}

		poly := physics.NewPolygonShape(physics.Transform{Position: pos, Rotation: rotation}, vertices)
		physicsSystem.AddShape(poly)
		velocities[poly.GetID()] = physics.Point{X: rand.Float64()*4 - 2, Y: rand.Float64()*4 - 2}
		angVelocities[poly.GetID()] = (rand.Float64()*2 - 1) * 0.1
	}

	game := &Game{
		physics:       physicsSystem,
		velocities:    velocities,
		angVelocities: angVelocities,
		screenWidth:   screenWidth,
		screenHeight:  screenHeight,
		gravity:       gravity,
	}

	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("Physlicks Example")
	if err := ebiten.RunGame(game); err != nil {
		panic(err)
	}
}
