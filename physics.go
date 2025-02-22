package gophyslicks

import (
	"image/color"
	"math"
	"sort"
)

// Point represents a 2D point or vector.
type Point struct {
	X, Y float64
}

// Add adds two points.
func (p Point) Add(q Point) Point {
	return Point{p.X + q.X, p.Y + q.Y}
}

// Sub subtracts q from p.
func (p Point) Sub(q Point) Point {
	return Point{p.X - q.X, p.Y - q.Y}
}

// Scale scales the point by a factor.
func (p Point) Scale(s float64) Point {
	return Point{p.X * s, p.Y * s}
}

// Dot computes the Dot product of two points.
func Dot(p, q Point) float64 {
	return p.X*q.X + p.Y*q.Y
}

// Magnitude returns the length of the vector.
func (p Point) Magnitude() float64 {
	return math.Sqrt(Dot(p, p))
}

// Normalize returns a unit vector in the same direction.
func (p Point) Normalize() Point {
	mag := p.Magnitude()
	if mag == 0 {
		return Point{0, 0}
	}
	return Point{p.X / mag, p.Y / mag}
}

// Perpendicular returns a perpendicular vector (rotated 90° counterclockwise).
func (p Point) Perpendicular() Point {
	return Point{-p.Y, p.X}
}

// AABB represents an axis-aligned bounding box.
type AABB struct {
	Min, Max Point
}

// Intersects checks if two AABBs intersect.
func (a AABB) Intersects(b AABB) bool {
	return a.Min.X < b.Max.X && a.Max.X > b.Min.X && a.Min.Y < b.Max.Y && a.Max.Y > b.Min.Y
}

// ShapeType identifies the type of shape.
type ShapeType int

const (
	// CircleType is circle shape
	CircleType ShapeType = iota
	// PolygonType is polygon shape
	PolygonType
)

// Shape is the interface for all collidable objects.
type Shape interface {
	GetAABB() AABB
	GetType() ShapeType
	GetID() int
	SetID(id int)
	Move(delta Point)
	GetColor() color.Color
	GetPosition() Point
	SetPosition(pos Point)
	GetWorldVertices() []Point   // For polygons
	GetRotation() float64        // Current rotation
	SetRotation(rot float64)     // Set rotation
	GetMass() float64            // Mass for physics
	GetElasticity() float64      // Coefficient of restitution
	GetMomentOfInertia() float64 // Rotational inertia
}

// BaseShape provides common fields for all shapes.
type BaseShape struct {
	ID            int
	Mass          float64 // Mass in arbitrary units
	Elasticity    float64 // Coefficient of restitution (0 = inelastic, 1 = fully elastic)
	MomentInertia float64 // Moment of inertia for rotation
}

// GetID returns the shape's ID.
func (b *BaseShape) GetID() int {
	return b.ID
}

// SetID sets the shape's ID.
func (b *BaseShape) SetID(id int) {
	b.ID = id
}

// GetMass returns the shape's mass.
func (b *BaseShape) GetMass() float64 {
	return b.Mass
}

// GetElasticity returns the shape's elasticity.
func (b *BaseShape) GetElasticity() float64 {
	return b.Elasticity
}

// GetMomentOfInertia returns the shape's moment of inertia.
func (b *BaseShape) GetMomentOfInertia() float64 {
	return b.MomentInertia
}

// CircleShape represents a circle with a position and radius.
type CircleShape struct {
	*BaseShape
	Position Point
	Radius   float64
	Rotation float64 // Not used for rendering circles but included for consistency
}

// NewCircleShape creates a new circle.
func NewCircleShape(position Point, radius float64) *CircleShape {
	mass := math.Pi * radius * radius // Mass proportional to area
	// Moment of inertia for a solid disk: I = (1/2) * m * r^2
	moment := 0.5 * mass * radius * radius
	return &CircleShape{
		BaseShape: &BaseShape{Mass: mass, Elasticity: 0.8, MomentInertia: moment},
		Position:  position,
		Radius:    radius,
		Rotation:  0,
	}
}

// GetType returns CircleType.
func (c *CircleShape) GetType() ShapeType {
	return CircleType
}

// GetAABB computes the AABB of the circle.
func (c *CircleShape) GetAABB() AABB {
	return AABB{
		Min: Point{c.Position.X - c.Radius, c.Position.Y - c.Radius},
		Max: Point{c.Position.X + c.Radius, c.Position.Y + c.Radius},
	}
}

// Move updates the position of the circle.
func (c *CircleShape) Move(delta Point) {
	c.Position = c.Position.Add(delta)
}

// GetColor returns a unique color based on the shape's ID.
func (c *CircleShape) GetColor() color.Color {
	return idToColor(c.ID)
}

// GetPosition returns the position of the circle.
func (c *CircleShape) GetPosition() Point {
	return c.Position
}

// SetPosition sets the position of the circle.
func (c *CircleShape) SetPosition(pos Point) {
	c.Position = pos
}

// GetWorldVertices returns an empty slice for circles.
func (c *CircleShape) GetWorldVertices() []Point {
	return []Point{}
}

// GetRotation returns the rotation (unused for circles).
func (c *CircleShape) GetRotation() float64 {
	return c.Rotation
}

// SetRotation sets the rotation (unused for circles but included for interface).
func (c *CircleShape) SetRotation(rot float64) {
	c.Rotation = rot
}

// Transform holds position and rotation data.
type Transform struct {
	Position Point
	Rotation float64 // in radians
}

// PolygonShape represents a convex polygon (e.g., rectangle or general polygon).
type PolygonShape struct {
	*BaseShape
	Transform     Transform
	LocalVertices []Point
}

// NewPolygonShape creates a new polygon.
func NewPolygonShape(transform Transform, localVerts []Point) *PolygonShape {
	// Calculate mass as area using the shoelace formula
	area := 0.0
	for i := 0; i < len(localVerts); i++ {
		j := (i + 1) % len(localVerts)
		area += localVerts[i].X*localVerts[j].Y - localVerts[j].X*localVerts[i].Y
	}
	area = math.Abs(area) / 2
	// Moment of inertia approximated for a polygon (simplified as a rectangle-like shape)
	// I = (m * (w^2 + h^2)) / 12, using bounding box dimensions
	minX, maxX := localVerts[0].X, localVerts[0].X
	minY, maxY := localVerts[0].Y, localVerts[0].Y
	for _, v := range localVerts[1:] {
		if v.X < minX {
			minX = v.X
		}
		if v.X > maxX {
			maxX = v.X
		}
		if v.Y < minY {
			minY = v.Y
		}
		if v.Y > maxY {
			maxY = v.Y
		}
	}
	w := maxX - minX
	h := maxY - minY
	moment := (area * (w*w + h*h)) / 12
	return &PolygonShape{
		BaseShape:     &BaseShape{Mass: area, Elasticity: 0.8, MomentInertia: moment},
		Transform:     transform,
		LocalVertices: localVerts,
	}
}

// NewRectShape creates a rotated rectangle as a polygon.
func NewRectShape(center Point, width, height, rotation float64) *PolygonShape {
	halfW, halfH := width/2, height/2
	localVerts := []Point{
		{-halfW, -halfH},
		{halfW, -halfH},
		{halfW, halfH},
		{-halfW, halfH},
	}
	return NewPolygonShape(Transform{center, rotation}, localVerts)
}

// NewTriangleShape creates a triangle as a polygon.
func NewTriangleShape(center Point, size float64, rotation float64) *PolygonShape {
	halfSize := size / 2
	localVerts := []Point{
		{0, -halfSize},        // Top
		{halfSize, halfSize},  // Bottom right
		{-halfSize, halfSize}, // Bottom left
	}
	return NewPolygonShape(Transform{center, rotation}, localVerts)
}

// GetType returns PolygonType.
func (p *PolygonShape) GetType() ShapeType {
	return PolygonType
}

// GetWorldVertices computes the vertices in world space.
func (p *PolygonShape) GetWorldVertices() []Point {
	worldVerts := make([]Point, len(p.LocalVertices))
	cos, sin := math.Cos(p.Transform.Rotation), math.Sin(p.Transform.Rotation)
	for i, v := range p.LocalVertices {
		rotatedX := v.X*cos - v.Y*sin
		rotatedY := v.X*sin + v.Y*cos
		worldVerts[i] = Point{rotatedX + p.Transform.Position.X, rotatedY + p.Transform.Position.Y}
	}
	return worldVerts
}

// GetAABB computes the AABB of the polygon.
func (p *PolygonShape) GetAABB() AABB {
	verts := p.GetWorldVertices()
	if len(verts) == 0 {
		return AABB{Point{0, 0}, Point{0, 0}}
	}
	minX, maxX := verts[0].X, verts[0].X
	minY, maxY := verts[0].Y, verts[0].Y
	for _, v := range verts[1:] {
		if v.X < minX {
			minX = v.X
		}
		if v.X > maxX {
			maxX = v.X
		}
		if v.Y < minY {
			minY = v.Y
		}
		if v.Y > maxY {
			maxY = v.Y
		}
	}
	return AABB{Point{minX, minY}, Point{maxX, maxY}}
}

// Move updates the position of the polygon.
func (p *PolygonShape) Move(delta Point) {
	p.Transform.Position = p.Transform.Position.Add(delta)
}

// GetColor returns a unique color based on the shape's ID.
func (p *PolygonShape) GetColor() color.Color {
	return idToColor(p.ID)
}

// GetPosition returns the position of the polygon.
func (p *PolygonShape) GetPosition() Point {
	return p.Transform.Position
}

// SetPosition sets the position of the polygon.
func (p *PolygonShape) SetPosition(pos Point) {
	p.Transform.Position = pos
}

// GetRotation returns the rotation of the polygon.
func (p *PolygonShape) GetRotation() float64 {
	return p.Transform.Rotation
}

// SetRotation sets the rotation of the polygon.
func (p *PolygonShape) SetRotation(rot float64) {
	p.Transform.Rotation = rot
}

// Quadtree represents a quadtree for spatial partitioning.
type Quadtree struct {
	bounds   AABB
	capacity int
	shapes   []Shape
	children []*Quadtree
	depth    int
	maxDepth int
}

// NewQuadtree initializes a quadtree node with depth tracking.
func NewQuadtree(bounds AABB, capacity int, depth int, maxDepth int) *Quadtree {
	return &Quadtree{
		bounds:   bounds,
		capacity: capacity,
		shapes:   []Shape{},
		children: []*Quadtree{},
		depth:    depth,
		maxDepth: maxDepth,
	}
}

// Insert adds a shape to the quadtree.
func (qt *Quadtree) Insert(shape Shape) {
	if !qt.bounds.Intersects(shape.GetAABB()) {
		return
	}
	if len(qt.children) == 0 && len(qt.shapes) < qt.capacity {
		qt.shapes = append(qt.shapes, shape)
		return
	}
	if len(qt.children) == 0 && qt.depth < qt.maxDepth {
		qt.subdivide()
	}
	if len(qt.children) > 0 {
		for _, child := range qt.children {
			child.Insert(shape)
		}
	} else {
		qt.shapes = append(qt.shapes, shape)
	}
}

// subdivide splits the node into four child quadtrees.
func (qt *Quadtree) subdivide() {
	midX := (qt.bounds.Min.X + qt.bounds.Max.X) / 2
	midY := (qt.bounds.Min.Y + qt.bounds.Max.Y) / 2
	qt.children = append(qt.children,
		NewQuadtree(AABB{Point{qt.bounds.Min.X, qt.bounds.Min.Y}, Point{midX, midY}}, qt.capacity, qt.depth+1, qt.maxDepth),
		NewQuadtree(AABB{Point{midX, qt.bounds.Min.Y}, Point{qt.bounds.Max.X, midY}}, qt.capacity, qt.depth+1, qt.maxDepth),
		NewQuadtree(AABB{Point{qt.bounds.Min.X, midY}, Point{midX, qt.bounds.Max.Y}}, qt.capacity, qt.depth+1, qt.maxDepth),
		NewQuadtree(AABB{Point{midX, midY}, Point{qt.bounds.Max.X, qt.bounds.Max.Y}}, qt.capacity, qt.depth+1, qt.maxDepth),
	)
	for _, shape := range qt.shapes {
		for _, child := range qt.children {
			child.Insert(shape)
		}
	}
	qt.shapes = []Shape{}
}

// Query finds all shapes that intersect with the given AABB.
func (qt *Quadtree) Query(aabb AABB) []Shape {
	var result []Shape
	if !qt.bounds.Intersects(aabb) {
		return result
	}
	for _, shape := range qt.shapes {
		if shape.GetAABB().Intersects(aabb) {
			result = append(result, shape)
		}
	}
	for _, child := range qt.children {
		result = append(result, child.Query(aabb)...)
	}
	return result
}

// PhysicsSystem manages shapes and collision detection.
type PhysicsSystem struct {
	shapes        []Shape
	quadtree      *Quadtree
	nextID        int
	worldSize     Point
	velocities    map[int]Point
	angVelocities map[int]float64
	gravity       Point
}

// NewPhysicsSystem initializes a physics system with a quadtree.
func NewPhysicsSystem(bounds AABB, capacity int, maxDepth int) *PhysicsSystem {
	return &PhysicsSystem{
		shapes:        []Shape{},
		quadtree:      NewQuadtree(bounds, capacity, 0, maxDepth),
		nextID:        0,
		worldSize:     Point{X: bounds.Max.X - bounds.Min.X, Y: bounds.Max.Y - bounds.Min.Y},
		velocities:    make(map[int]Point),
		angVelocities: make(map[int]float64),
		gravity:       Point{X: 0, Y: 0},
	}
}

// Update updates the physics system state including gravity, positions, rotations, and collisions.
func (ps *PhysicsSystem) Update() {
	// Apply gravity to velocities
	for _, shape := range ps.shapes {
		vel := ps.velocities[shape.GetID()]
		ps.velocities[shape.GetID()] = vel.Add(ps.gravity)
	}

	// Update positions and rotations
	for _, shape := range ps.shapes {
		vel := ps.velocities[shape.GetID()]
		angVel := ps.angVelocities[shape.GetID()]
		shape.Move(vel)
		shape.SetRotation(shape.GetRotation() + angVel)

		// Clamp to screen boundaries
		aabb := shape.GetAABB()
		if aabb.Min.X < 0 {
			shape.Move(Point{X: -aabb.Min.X, Y: 0})
			ps.velocities[shape.GetID()] = Point{X: -vel.X * shape.GetElasticity(), Y: vel.Y}
			ps.angVelocities[shape.GetID()] = -angVel * shape.GetElasticity()
		}
		if aabb.Max.X > ps.worldSize.X {
			shape.Move(Point{X: ps.worldSize.X - aabb.Max.X, Y: 0})
			ps.velocities[shape.GetID()] = Point{X: -vel.X * shape.GetElasticity(), Y: vel.Y}
			ps.angVelocities[shape.GetID()] = -angVel * shape.GetElasticity()
		}
		if aabb.Min.Y < 0 {
			shape.Move(Point{X: 0, Y: -aabb.Min.Y})
			ps.velocities[shape.GetID()] = Point{X: vel.X, Y: -vel.Y * shape.GetElasticity()}
			ps.angVelocities[shape.GetID()] = -angVel * shape.GetElasticity()
		}
		if aabb.Max.Y > ps.worldSize.Y {
			shape.Move(Point{X: 0, Y: ps.worldSize.Y - aabb.Max.Y})
			ps.velocities[shape.GetID()] = Point{X: vel.X, Y: -vel.Y * shape.GetElasticity()}
			ps.angVelocities[shape.GetID()] = -angVel * shape.GetElasticity()
		}
	}

	// Handle collisions
	collidingPairs := ps.DetectCollisions()
	for _, pair := range collidingPairs {
		resolveCollision(pair[0], pair[1], ps.velocities, ps.angVelocities)
	}
}

// GetShapes returns shapes in the system.
func (ps *PhysicsSystem) GetShapes() []Shape {
	return ps.shapes
}

// AddShape adds a shape to the system with initial velocity and angular velocity.
func (ps *PhysicsSystem) AddShape(s Shape, velocity Point, angularVelocity float64) {
	s.SetID(ps.nextID)
	ps.nextID++
	ps.shapes = append(ps.shapes, s)
	ps.velocities[s.GetID()] = velocity
	ps.angVelocities[s.GetID()] = angularVelocity
}

// DetectCollisions finds all colliding pairs using the quadtree.
func (ps *PhysicsSystem) DetectCollisions() [][]Shape {
	ps.quadtree = NewQuadtree(ps.quadtree.bounds, ps.quadtree.capacity, 0, ps.quadtree.maxDepth)
	for _, shape := range ps.shapes {
		ps.quadtree.Insert(shape)
	}
	collidingPairs := [][]Shape{}
	checked := make(map[[2]int]bool)
	for _, shape1 := range ps.shapes {
		potentialColliders := ps.quadtree.Query(shape1.GetAABB())
		for _, shape2 := range potentialColliders {
			if shape1.GetID() < shape2.GetID() && !checked[[2]int{shape1.GetID(), shape2.GetID()}] {
				checked[[2]int{shape1.GetID(), shape2.GetID()}] = true
				if Collide(shape1, shape2) {
					collidingPairs = append(collidingPairs, []Shape{shape1, shape2})
				}
			}
		}
	}
	return collidingPairs
}

// DetectCollisionsSpatialSort uses sweep and prune algorithm before quadtree
func (ps *PhysicsSystem) DetectCollisionsSpatialSort() [][]Shape {
	// Sort shapes by min X coordinate for sweep and prune
	type ShapeInterval struct {
		min, max float64
		shape    Shape
	}

	intervals := make([]ShapeInterval, len(ps.shapes))
	for i, shape := range ps.shapes {
		aabb := shape.GetAABB()
		intervals[i] = ShapeInterval{aabb.Min.X, aabb.Max.X, shape}
	}

	// Sort by min X
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i].min < intervals[j].min
	})

	// Find potentially colliding pairs using sweep and prune
	potentialPairs := make(map[[2]int]bool)
	activeIntervals := make([]ShapeInterval, 0)

	for _, interval := range intervals {
		// Remove inactive intervals
		newActive := make([]ShapeInterval, 0)
		for _, active := range activeIntervals {
			if active.max >= interval.min {
				newActive = append(newActive, active)
				// Add potential pair
				if active.shape.GetID() < interval.shape.GetID() {
					potentialPairs[[2]int{active.shape.GetID(), interval.shape.GetID()}] = true
				} else {
					potentialPairs[[2]int{interval.shape.GetID(), active.shape.GetID()}] = true
				}
			}
		}
		activeIntervals = append(newActive, interval)
	}

	// Use quadtree for refined check of potential pairs
	ps.quadtree = NewQuadtree(ps.quadtree.bounds, ps.quadtree.capacity, 0, ps.quadtree.maxDepth)
	for _, shape := range ps.shapes {
		ps.quadtree.Insert(shape)
	}

	// Final collision check
	collidingPairs := make([][]Shape, 0)
	for pair := range potentialPairs {
		shape1 := ps.getShapeByID(pair[0])
		shape2 := ps.getShapeByID(pair[1])
		if shape1 != nil && shape2 != nil && Collide(shape1, shape2) {
			collidingPairs = append(collidingPairs, []Shape{shape1, shape2})
		}
	}

	return collidingPairs
}

func (ps *PhysicsSystem) getShapeByID(id int) Shape {
	for _, shape := range ps.shapes {
		if shape.GetID() == id {
			return shape
		}
	}
	return nil
}

// resolveCollision resolves the collision with impulse-based physics including rotation.
func resolveCollision(shape1, shape2 Shape, velocities map[int]Point, angVelocities map[int]float64) {
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
	normal := GetCollisionNormal(shape1, shape2)

	// Relative velocity at contact point, including rotational effects
	v1AtContact := v1.Add(r1.Perpendicular().Scale(w1))
	v2AtContact := v2.Add(r2.Perpendicular().Scale(w2))
	rv := v2AtContact.Sub(v1AtContact)
	vn := Dot(rv, normal)

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
	OverlapAdjustment(shape1, shape2, normal)
}

// OverlapAdjustment adjusts positions to prevent overlap.
func OverlapAdjustment(shape1, shape2 Shape, normal Point) {
	const slop = 0.005
	const percent = 0.4
	aabb1 := shape1.GetAABB()
	aabb2 := shape2.GetAABB()
	overlapX := math.Min(aabb1.Max.X, aabb2.Max.X) - math.Max(aabb1.Min.X, aabb2.Min.X)
	overlapY := math.Min(aabb1.Max.Y, aabb2.Max.Y) - math.Max(aabb1.Min.Y, aabb2.Min.Y)
	if overlapX > 0 && overlapY > 0 {
		penetration := math.Min(overlapX, overlapY) - slop
		if penetration > 0 {
			correction := normal.Scale(penetration * percent)
			totalMass := shape1.GetMass() + shape2.GetMass()
			if totalMass > 0 {
				if shape1.GetMass() > 0 {
					shape1.SetPosition(shape1.GetPosition().Sub(correction.Scale(shape2.GetMass() / totalMass)))
				}
				if shape2.GetMass() > 0 {
					shape2.SetPosition(shape2.GetPosition().Add(correction.Scale(shape1.GetMass() / totalMass)))
				}
			}
		}
	}
}

// Collide dispatches to the appropriate collision function based on shape types.
func Collide(a, b Shape) bool {
	switch a.GetType() {
	case CircleType:
		switch b.GetType() {
		case CircleType:
			return circleCircleCollision(a.(*CircleShape), b.(*CircleShape))
		case PolygonType:
			return circlePolygonCollision(a.(*CircleShape), b.(*PolygonShape))
		}
	case PolygonType:
		switch b.GetType() {
		case CircleType:
			return circlePolygonCollision(b.(*CircleShape), a.(*PolygonShape))
		case PolygonType:
			return polygonPolygonCollision(a.(*PolygonShape), b.(*PolygonShape))
		}
	}
	return false
}

// circleCircleCollision checks if two circles intersect.
func circleCircleCollision(c1, c2 *CircleShape) bool {
	distSquared := Dot(c1.Position.Sub(c2.Position), c1.Position.Sub(c2.Position))
	radiusSum := c1.Radius + c2.Radius
	return distSquared < radiusSum*radiusSum
}

// circlePolygonCollision checks if a circle and a polygon intersect.
func circlePolygonCollision(circle *CircleShape, poly *PolygonShape) bool {
	verts := poly.GetWorldVertices()
	if isPointInsideConvexPolygon(circle.Position, verts) {
		return true
	}
	for i := 0; i < len(verts); i++ {
		A := verts[i]
		B := verts[(i+1)%len(verts)]
		closest := closestPointOnSegment(A, B, circle.Position)
		vec := closest.Sub(circle.Position)
		if Dot(vec, vec) < circle.Radius*circle.Radius {
			return true
		}
	}
	return false
}

// polygonPolygonCollision uses SAT to check if two polygons intersect.
func polygonPolygonCollision(poly1, poly2 *PolygonShape) bool {
	axes := getSeparatingAxes(poly1, poly2)
	for _, axis := range axes {
		min1, max1 := projectPolygon(poly1, axis)
		min2, max2 := projectPolygon(poly2, axis)
		if !intervalsOverlap(min1, max1, min2, max2) {
			return false
		}
	}
	return true
}

// closestPointOnSegment finds the closest point on a line segment to a point.
func closestPointOnSegment(A, B, P Point) Point {
	AB := B.Sub(A)
	AP := P.Sub(A)
	dotAPAB := Dot(AP, AB)
	lenSquared := Dot(AB, AB)
	if lenSquared == 0 {
		return A
	}
	t := dotAPAB / lenSquared
	if t < 0 {
		return A
	} else if t > 1 {
		return B
	}
	return A.Add(AB.Scale(t))
}

// isPointInsideConvexPolygon checks if a point is inside a convex polygon.
func isPointInsideConvexPolygon(P Point, verts []Point) bool {
	if len(verts) < 3 {
		return false
	}
	for i := 0; i < len(verts); i++ {
		A := verts[i]
		B := verts[(i+1)%len(verts)]
		AB := B.Sub(A)
		AP := P.Sub(A)
		cross := AB.X*AP.Y - AB.Y*AP.X
		if cross <= 0 {
			return false
		}
	}
	return true
}

// getSeparatingAxes returns the potential separating axes for SAT.
func getSeparatingAxes(poly1, poly2 *PolygonShape) []Point {
	axes := []Point{}
	verts1 := poly1.GetWorldVertices()
	verts2 := poly2.GetWorldVertices()
	for i := 0; i < len(verts1); i++ {
		A := verts1[i]
		B := verts1[(i+1)%len(verts1)]
		edge := B.Sub(A)
		axes = append(axes, Point{edge.Y, -edge.X})
	}
	for i := 0; i < len(verts2); i++ {
		A := verts2[i]
		B := verts2[(i+1)%len(verts2)]
		edge := B.Sub(A)
		axes = append(axes, Point{edge.Y, -edge.X})
	}
	return axes
}

// projectPolygon projects a polygon onto an axis.
func projectPolygon(poly *PolygonShape, axis Point) (min, max float64) {
	verts := poly.GetWorldVertices()
	min = Dot(verts[0], axis)
	max = min
	for _, v := range verts[1:] {
		proj := Dot(v, axis)
		if proj < min {
			min = proj
		}
		if proj > max {
			max = proj
		}
	}
	return min, max
}

// intervalsOverlap checks if two intervals overlap.
func intervalsOverlap(min1, max1, min2, max2 float64) bool {
	return !(max1 < min2 || max2 < min1)
}

// idToColor generates a unique color based on the shape's ID.
func idToColor(id int) color.Color {
	r := uint8((id * 37) % 256)
	g := uint8((id * 53) % 256)
	b := uint8((id * 101) % 256)
	return color.RGBA{r, g, b, 255}
}

// GetCollisionNormal computes the collision normal based on shape types.
func GetCollisionNormal(shape1, shape2 Shape) Point {
	// Circle-Circle collision
	if shape1.GetType() == CircleType && shape2.GetType() == CircleType {
		c1 := shape1.(*CircleShape)
		c2 := shape2.(*CircleShape)
		normal := c2.Position.Sub(c1.Position)
		if normal.Magnitude() == 0 {
			return Point{X: 1, Y: 0}
		}
		return normal.Normalize()
	}

	// Circle-Polygon collision
	if shape1.GetType() == CircleType && shape2.GetType() == PolygonType {
		circle := shape1.(*CircleShape)
		poly := shape2.(*PolygonShape)
		verts := poly.GetWorldVertices()
		minDist := math.Inf(1)
		var bestNormal Point
		for i := 0; i < len(verts); i++ {
			A := verts[i]
			B := verts[(i+1)%len(verts)]
			edge := B.Sub(A)
			// Outward normal for the edge (assuming clockwise winding)
			edgeNormal := Point{X: edge.Y, Y: -edge.X}.Normalize()
			projection := Dot(circle.Position.Sub(A), edgeNormal)
			if projection < minDist {
				minDist = projection
				bestNormal = edgeNormal
			}
		}
		return bestNormal
	}

	// Polygon-Circle collision (swap and invert normal)
	if shape1.GetType() == PolygonType && shape2.GetType() == CircleType {
		normal := GetCollisionNormal(shape2, shape1)
		return normal.Scale(-1)
	}

	// Polygon-Polygon collision using SAT
	if shape1.GetType() == PolygonType && shape2.GetType() == PolygonType {
		poly1 := shape1.(*PolygonShape)
		poly2 := shape2.(*PolygonShape)
		axes := getSeparatingAxes(poly1, poly2)
		minOverlap := math.Inf(1)
		var minAxis Point
		for _, axis := range axes {
			min1, max1 := projectPolygon(poly1, axis)
			min2, max2 := projectPolygon(poly2, axis)
			overlap := math.Min(max1, max2) - math.Max(min1, min2)
			if overlap < 0 {
				return Point{} // No collision, but shouldn't happen here
			}
			if overlap < minOverlap {
				minOverlap = overlap
				minAxis = axis
			}
		}
		// Determine normal direction
		center1 := poly1.GetPosition()
		center2 := poly2.GetPosition()
		delta := center2.Sub(center1)
		if Dot(delta, minAxis) < 0 {
			minAxis = minAxis.Scale(-1)
		}
		return minAxis.Normalize()
	}

	// Default case
	return Point{X: 1, Y: 0}
}
