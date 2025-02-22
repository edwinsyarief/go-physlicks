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
	physics      *physics.PhysicsSystem
	screenWidth  int
	screenHeight int
}

// Update updates the game state.
func (g *Game) Update() error {
	g.physics.Update()
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

// Layout sets the screen dimensions.
func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	return g.screenWidth, g.screenHeight
}

func drawCircle(screen *ebiten.Image, x, y, radius float64, col color.Color) {
	const segments = 32
	for i := 0; i < segments; i++ {
		angle1 := float64(i) * 2 * math.Pi / segments
		angle2 := float64(i+1) * 2 * math.Pi / segments
		x1 := x + radius*math.Cos(angle1)
		y1 := y + radius*math.Sin(angle1)
		x2 := x + radius*math.Cos(angle2)
		y2 := y + radius*math.Sin(angle2)
		ebitenutil.DrawLine(screen, x1, y1, x2, y2, col)
	}
}

func drawPolygon(screen *ebiten.Image, vertices []physics.Point, col color.Color) {
	for i := 0; i < len(vertices); i++ {
		j := (i + 1) % len(vertices)
		ebitenutil.DrawLine(screen, vertices[i].X, vertices[i].Y, vertices[j].X, vertices[j].Y, col)
	}
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
	physicsSystem := physics.NewPhysicsSystem(bounds, quadtreeCapacity, quadtreeMaxDepth)

	// Add circles
	for i := 0; i < numCircles; i++ {
		pos := physics.Point{X: rand.Float64() * screenWidth, Y: rand.Float64() * screenHeight}
		radius := 5 + rand.Float64()*15
		circle := physics.NewCircleShape(pos, radius)
		velocity := physics.Point{X: rand.Float64()*4 - 2, Y: rand.Float64()*4 - 2}
		angVel := (rand.Float64()*2 - 1) * 0.1
		physicsSystem.AddShape(circle, velocity, angVel)
	}

	// Add rectangles
	for i := 0; i < numRects; i++ {
		pos := physics.Point{X: rand.Float64() * screenWidth, Y: rand.Float64() * screenHeight}
		width := 10 + rand.Float64()*30
		height := 10 + rand.Float64()*30
		rotation := rand.Float64() * 2 * math.Pi
		rect := physics.NewRectShape(pos, width, height, rotation)
		velocity := physics.Point{X: rand.Float64()*4 - 2, Y: rand.Float64()*4 - 2}
		angVel := (rand.Float64()*2 - 1) * 0.1
		physicsSystem.AddShape(rect, velocity, angVel)
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
		velocity := physics.Point{X: rand.Float64()*4 - 2, Y: rand.Float64()*4 - 2}
		angVel := (rand.Float64()*2 - 1) * 0.1
		physicsSystem.AddShape(poly, velocity, angVel)
	}

	game := &Game{
		physics:      physicsSystem,
		screenWidth:  screenWidth,
		screenHeight: screenHeight,
	}

	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("Physlicks Example")
	if err := ebiten.RunGame(game); err != nil {
		panic(err)
	}
}
