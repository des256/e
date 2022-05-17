# E - UI

Shamelessly steal from Flutter architecture, but dump Dart and some Googlisms.

## Framework

### Main Loop

Use Tokio.

### Pipeline

1. user input: respond to input gestures
2. animation: timed UI changes
3. build: (re)build widgets from app code
4. layout: positioning and sizing of elements on the screen
5. paint: convert elements into visual representation
6. composition: overlay visual elements in draw order
7. rasterize: translate output into GPU instructions

### Trees

re-use elements when updating the tree

#### Widgets -> Configs

describes configuration of an element

hold properties
offer public API

Padding: hold onto child and amount of padding

#### Elements -> Views

instantiation of widgets, manages the lifecycle

holds spot in UI hierarchy
manage parent/child relationship

#### Render Objects -> Primitives

handles size, layout and painting

size/paint
layout children
claim input events?

Padding: set size to child + padding and offset child

### Layout

1. pass down box constraints to the children
2. children pass up their size back to the parents

render tree node does not know its position, only size

flex layout
relayout boundaries:
    - decide by parent (tight constraints)
    - if parent doesn't care about child (like scrolling views)
    - if child uses the parent's constraints, but not ask any children

order:
    first visit inflexible children
    then visit flexible children

### Painting

walk tree in depth order
deposit paint commands in different layers
children decide into which layer they should be drawn, so parent continues drawing in that layer

repaint boundaries:
    maybe built-in into widget design

### Compositing

composited scrolling: each item has their own layer, and only need to move, they are offset-independent, so there is optimal re-use of everything

layers are display lists or rendered textures

texturization when drawn display list 3 times
except:
    - when it's really simple (just a rectangle)
    - when ther are mostly transparent pixels

### Debugging

keep statistics on repaint boundaries

### Design Rules

### Widgets

reactive model, stateless/stateful, connect to rendering

### Rendering

abstractions dealing with layout

widgets are expressed in element tree based on components and render objects

### Animation, Painting and Gestures

### Foundation

abstractions over animation, painting and gestures

## Engine

to connect to parts of the engine, use binding structures that allow for all sorts of debugging and analysis

### Composition

### Rendering

### System Events

### Asset Resolution

### Frame Scheduling

### Frame Pipeline

### Text Layout

## Embedder

### Render Surface Setup

### Thread Setup

### Event Loop Interop

### App Packaging
