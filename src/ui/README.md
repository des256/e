# E - UI

Shamelessly steal from Flutter architecture, but dump Dart and some Googlisms.

## Framework

### Main Loop

Use Tokio.

### Pipeline

Flutter has 3 trees: WidgetTree -> ElementTree -> RenderObjectTree

<Widget>: The configuration of the ElementTree.
<Element>: The user interface.
<RenderObject>: What actually gets rendered in the end.

Flutter rendering pipeline: UI: [ Animate -> Build -> Layout -> Paint -> Submit ] -> LayerTree -> GPU: [ Rasterize -> Composite ]

For user input and description, use Xelim-like function as Widget structure, including callbacks. This function describes the configuration of the UI elements.

Whenever something changes, use setState() to propagate the change to the UI. This triggers calling of build() again, which creates a new Widget. From this new Widget, createElement or updateElement are then called to keep up the Element tree. Finally, createRenderObject or updateRenderObject gets called accordingly.


1. User Input on the currently visible rasterized UI
2. Animation, changes due to timers
3. Build, not entirely sure...
4. Layout, figure out where the widget can render itself
5. Paint, generate the rectangles to be composited (GPU)
6. Composition, generate command buffer in proper draw order (GPU)
7. Rasterize, render frame (GPU)

1. user input: respond to input gestures

    Ultimately come from the system and refer to what the previous layout resulted in
    This is also where the tree is modified

    self = self.handle_event(event: &Event)

2. animation: timed UI changes, similar result as 1.
3. build: (re)build widgets from app code
4. layout: positioning and sizing of elements on the screen

    Can be tiny adjustments, also modifying the tree

    elements = self.animate_build_layout()

5. paint (GPU): convert elements into visual representation

    bunch of mostly reused/unchanged rectangles with pixels = elements.paint(previous bunch of reused/unchanged rectangles with pixels)

6. composition (GPU): overlay visual elements in draw order

    reordered bunch of reused/unchanged rectangles with pixels = bunch.compose()

7. rasterize (GPU): translate output into GPU instructions

    reordered.render()

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
