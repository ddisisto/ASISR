# Why Deep Learning Works Unreasonably Well
(youtube)
Welch Labs
757K subscribers  
146,611 views  10 Aug 2025


In 1989, George Sabeno proved what's now known as the universal approximation
theorem. If we take some complex function, for example, this really complicated border in the town of Barlay
Hertok, these parts of the map are in Belgium and these parts are in the Netherlands. The universal approximation
theorem guarantees that there exists a two-layer neural network that can fit this border as precisely as we want. A
nice way to get a feel for this result is to see what a two-layer network like this does. geometrically.
Most modern neural networks use some version of rectified linear activation functions. Visually, this means that
each neuron in the first layer of our network folds up a copy of our map along a single fold line where the location of
the fold line is controlled by the neurons learned weights. From here, our first neuron in our second layer takes
in these bent planes and multiplies their heights by another learned weight value, which geometrically further bends
up or down the folded parts of our planes and flips over our folded region when that neuron's weight value is
negative. These three bent planes are then added together by our neuron, resulting in a surface like this. Our
three-fold lines from our first layer now divide up our map into these five regions that each become different
planes in our second layer surface. This surface shows the output of our first neuron in our second layer. The second
neuron in our second layer flips, scales, and combines our first planes using different learned parameters,
resulting in this surface that again uses the same five regions of our map, but at different heights.
The height of the surface formed by our first neuron corresponds to the model's confidence in a certain part of the map
being in the Netherlands. And the height of the second neuron surface corresponds to the model's confidence in Belgium.
Coloring our Netherlands surface blue and our Belgium surface yellow. And bringing these surfaces together onto
the same axis, the intersection of our surfaces shows us where our model is equally confident in both countries.
This is the model's learn decision boundary which gives us a basic border
separating the core Belgium region from the surrounding Netherlands region. Now the universal approximation theorem
tells us that if we just keep adding neurons to our first layer, eventually we'll land on an architecture capable of
representing our full border. Training a network with eight neurons in its first layer, we get this set of eight folds
leading to this surface for our first output neuron and this surface for our second output neuron. Bringing these new
surfaces onto the same axis, we see these new more complex intersection lines leading to this more detailed
final border that begins to break our map into separate regions. Here's the surfaces and border for a
larger model with 16 neurons. Here's a 32 neuron model. Here's 64. And here's
128. It starts to become difficult to see how our surfaces are intersecting exactly
with this many fold lines. Let's flatten out our surfaces to make it easier to see how the model uses all its different
fold lines to fit our border. Doubling our neuron count again to 256.
Here's how our neurons divide up our map. And here's the final decision boundary. Here's 512 neurons. And here
is 1,024. We're getting closer to our true border, but we're still missing a number of
parts of the town. And we've reached a point where I can't actually render any more polygons, but we can still render
our decision boundary. Here's the border we get with a 10,000 neuron model. And
finally, here's a model with a 100,000 neurons. We're getting even closer at this point,
but even with a 100,000 neurons, there's a couple of parts of the border that our model hasn't learned.
It feels like the universal approximation theorem isn't really working. What are we missing here? But
before we get into the details of what's going wrong, let me show you one more thing. Let's take just 128 neurons. But
instead of arranging them in a single wide layer, let's arrange them in four separate layers of 32 each like this,
where the output of each layer is passed into the next. After training this model, these are the resulting learned
regions and decision boundary. Our five layer network with just 130 total
neurons is able to learn a more precise border than our 100,000 neuron model and
it's able to divide up our map more effectively. How is it that rearranging our neurons
into multiple layers makes our model so much more powerful? The neurons in both our deeper and shallow models do the
same folding, scaling, and combining operations. Why are these operations so much more
effective when composed in multiple layers? And how does the geometry of our map change as it moves through these
stacked operations? This video is part of a series sponsored

~~sponsor segment~~

Part 2 Recap
Last time in part two of this series, we dug into the mathematics of how modern models are trained using back
propagation and gradient descent. We saw how given the inputs of latitude and longitude, a single layer model can
effectively learn to position planes over different European cities, learning to separate Paris, Berlin, Madrid, and
Barcelona. The key piece of functionality here is that our model learns to position the Madrid plane
above all the other planes above Madrid, the Barcelona plane above all the other planes above Barcelona, and so on. And
the height of our planes corresponds to our network's final confidence in a specific city. We left off considering
the most complex geographic border in the world between Belgium and the Netherlands in the municipality of
Barlay Herto. Given a single plane for each country, there's no way to position our planes.
So our Belgium plane is on top of our Netherlands plane above only the Belgium portions of our map. Another way to
think about this is that our two tilted planes intersect at a line on our map where everything on one side of the line
will be classified as part of Belgium and everything on the other side will be classified as part of the Netherlands.
And there's no way this linear decision boundary can correctly divide up our city. Our networks from last time looked
like this with a single layer of neurons between our inputs and softmax function.
As we saw last time, the softmax function bends our planes to output nice final probability values. But
importantly, it doesn't change the location of the decision boundaries at the intersections of our planes. For
this reason, we won't concern ourselves too much with softmax in this video. The networks we saw at the beginning of this
Moving to Two Layers
video add one more layer of neurons and are able to accomplish significantly more. Just like the neurons in our
simple single layer model, each of the neurons in the first layer of our two-layer network contains a simple
linear model that geometrically looks like a plane. Mathematically, the first
neuron in our first layer takes in the coordinates of our point, multiplies each coordinate by a learned number
called a weight, which we're writing here using lowercase m, and adds these results together. The weight values
control the steepness of our plane in each direction. Finally, we add one more learnable parameter called a bias. This
shifts our whole plane up and down. So, if we pass in this Belgium point on our
map with coordinates of x1= 0.6 and x2= 0.4, we multiply our x1 value by our
first weight and our x2 value by our second weight and add these results together. And finally, we add our bias
value to compute our final result of minus0.14. This computed value corresponds to the
height of our first neurons plane at these input coordinates. Now, if we just pass the height of our plane, minus0.14
How Activation Functions Fold Space
in this example, into our second layer of neurons, our multiple layers of neurons will actually just collapse back
down into what is effectively a single layer of neurons. We can show this collapsing algebraically. There's just a
bunch of terms to deal with. These first two equations correspond to the first two neurons in our first layer. Note
that we're using these superscripts to keep track of where each weight comes from. Everything with a superscript of
one comes from our model's first layer. Here's the equation for the first neuron in our second layer. If we pass the
outputs of our first layer directly into our second layer, this is equivalent to plugging in our first set of equations
into our second equation like this. Distributing and collecting terms, we
end up with a new constant times our input x1 plus another new constant times our input x2 plus this final constant.
This equation has the same shape as our individual neuron equations just with different constants.
This result tells us that if we just hook up the outputs of one layer of plane fitting neurons to the inputs of
our next layer, we end up adding together different tilted planes, which just results in a different tilted
plane. So, a two-layer network connected like this is still only capable in practice of fitting two planes to our
map, just as our single layer model did. For our multi-layer neural network to be
able to learn more complex patterns, we need to add one more small piece of math.
We'll pass the output of our planes from our first layer into a function called an activation function that will modify
their shape into something more complex for our model to work with. It turns out that we can build high performing neural
networks using a variety of activation functions. But one of the simplest and most widely used today is a function
called a rectified linear unit or RLU. RLU is incredibly simple. For input
values less than zero, RLU returns zero. And for input values greater than or equal to zero, ru simply passes its
input value through. So ru of minus1 is zero and ru of one is one.
Applying our ru activation function to the output planes of our first layer. The regions of our plane with heights
less than zero are folded up or clipped to a height of zero. So instead of outputting planes, the first layer of
our network now outputs these bent planes. This is the folding operation we saw at the beginning of the video. So to
Numerical Walkthrough
decide which country a point is in. For example, this Belgium point with coordinates of 0.60.4 that we saw
earlier, we pass these coordinates into our first layer of neurons and get values of minus0.14 and minus0.33 out
corresponding to the height of each of our planes at the input coordinates of our point. From here we apply our ReLU
activation function folding all values below zero up to zero. The height of our
point on both planes is negative. So we set both points to zero. So our input
point 0.60.4 has now been mapped to values of 0 0 by our bent planes. From here our final
layer of neurons multiplies these values of 0 by its weights and adds its bias
terms, shifting our combined bent planes up and down. moving our point to 0.03 on
our top surface and minus 0.89 on our bottom surface. Our second neuron's
output corresponds to our model's confidence in Belgium, which is higher in this case, meaning this point will be
classified as being in Belgium. And visually, we see this point on our Belgium plane being above our point on
our Netherlands plane. We can do a similar analysis for a point in the Netherlands like this point at 0.3 0.7.
The key difference here is that this point does not fall in the zero relu region of our second neuron. Meaning
that it gets pushed up on our final Netherlands bent plane and pushed down on our final Belgium plane, resulting in
a correct Netherlands classification. So our bent plane geometry is equivalent
numerically to moving data through our network, but gives us a nice way to see how all of the points on our map are
processed at once. As we add more neurons to our first layer, we're able to make more and more
folds in our map, cutting our map into more and more regions for our output neurons to push up and down into more
and more complex surfaces. Now, as we saw at the opening of the video, assuming a sufficient number of
Universal Approximation Theorem
neurons in our first layer, the universal approximation theorem tells us that a two-layer neural network exists
that can represent the borders of our town at arbitrarily high precision. But as we saw, even at a 100,000 neurons
in our first layer, we were not able to successfully train a model to completely match our borders. What is going on
here? The universal approximation theorem is sometimes mistaken to mean
that neural networks can learn anything. But what it really says is that a wide enough neural network is capable of
representing any continuous function. Now the borders of our town are actually
not continuous. But the continuity that the theorem is referring to here is actually the continuity of the final
surfaces that we intersect to find our border. The real issue here is that
although the universal approximation theorem tells us that a two-layer solution exists, it does not mean that
in practice we can actually find the solution. And the theorem does not tell us how many neurons we actually need to
solve a given problem. As we saw in parts one and two of this series, modern neural networks learn
using back propagation and gradient descent, which provide no guarantees of finding the best or even a good
solution. Instead, these algorithms make small iterative updates to our parameters, and we typically just stop
the learning process when performance stops improving. Before training, our network is randomly
initialized, placing our fold lines at random locations on our map. Here's one
initialization for our five neuron two-layer model. Here's how our folded planes are combined by the second layer
of this model. And here's how these surfaces intersect to form a decision boundary before training. If we pass in
the Belgium point we considered earlier into our randomly initialized model, this point ends up on this planer region
in our second layer surface. This first neuron surface ends up on top in our
final output shown here in blue. meaning that our model incorrectly classifies our point as being in the Netherlands.
The Geometry of Backpropagation
This error is measured using the cross entropy loss as we saw in part two. And this loss is then run through our back
propagation algorithm resulting in gradient values for each of our model 17 parameters.
Some of the largest resulting gradients are for this third neuron in our first layer. Both our DLDM31 and our DLDB3
gradients are large and negative. Currently m31 is negative tilting our
plane down in the x1 direction. Our gradient is telling us that to decrease
our loss we should increase m31 which will reduce the slope of our plane
making it flatter. Back propagation also returns a large negative value for dlddb3
which tells us to shift our whole plane upwards. Adjusting our parameters in this
direction moves our plane and shifts our RLU joint line to the right.
Zooming out to our full network, we can see how this update moves the center fold line in our second layer to the
right. On our final surfaces, our update moves our top blue surface down,
reducing the model's confidence in the incorrect answer of the Netherlands while moving our decision boundary to
the right. We can now repeat this gradient descent process and watch our model learn. Step
by step, these small updates adjust both the locations of the fold lines in our first layer and the way these bent
planes are combined by our second layer until we have a nice concave down surface on top of Belgium that
intersects a concave up Netherland surface at a nice border. Now, when I initially tried to train
this model, it didn't actually work nearly this well. I had a different random initialization that looked more
like this. Placing our blue surface on top of our yellow surface when we want
our model to learn the exact opposite orientation with a central yellow region for Belgium on top. As our model learns
from this starting point, our back propagation algorithm begins to reverse the orientation of these surfaces,
lowering our loss values and moving the blue surface down and the yellow surface up. But in doing so, back propagation
pushes the decision boundaries off of our planes, leaving our whole town in the zeroed out part of our bent relu
plane. Gradient descent is not able to recover from this configuration since
the gradients through the zeroed out part of our RLU activation function are also zero, leaving our model with
effectively a single plane to work with, resulting in a sub-optimal linear decision boundary. So even though we
know that a nice solution exists for our five neuron network given this starting point gradient descent is not able to
find it in the case of our super wide 100,000 neuron network there may be
analogously good solutions out there we just can't reach them with gradient descent
now there is some subtlety here as we saw back in part one when models become large the chances of gradient descent
actually getting stuck in a local minimum in this highdimensional loss landscape becomes very small. Our super
wide network is probably not getting stuck in quite the same way as our small network.
In addition to not telling us how to find a specific solution, the universal approximation theorem also does not tell
us how many neurons we actually need to solve a given problem. And in fact, for a broad class of functions, it's been
shown that the number of neurons we need in a shallow network is exponentially larger than the number of neurons needed
in a deep network. So, it's possible that a 100,000 neurons may actually not
be enough. Finally, it's difficult to prove a negative. In the course of making this video, I experimented with a
bunch of different optimizer configurations for these wide models. But it wouldn't surprise me if there's a
way to train a 100,000 neuron, a 10,000 neuron, or maybe even smaller two-layer model to fit the borders of our town.
I'll leave a link to my code in the description if you want to experiment. And please send me your results if you make progress. I would love to see a
The Geometry of Depth
solution. Exact number of neurons aside, as we saw earlier, we can make incredible efficiency gains by going
deep instead of wide, stacking our neurons into additional layers. And that's where we'll turn our attention
next. What new geometry does stacking our layers create? And how does this
geometry help our model learn the complex borders of our town? Let's begin with a simple two-layer model with two
neurons each. This simple two-layer model learns these folds in our map which are combined by our second layer
into this bent up surface and this bent down surface. Taking the intersection of our surfaces where our model is equally
confident in both countries, we get this simple decision boundary. Now let's add
a third layer to our model with two additional neurons. So we now have three layers and six neurons total. After
training, our first layer learns to fold our input planes like this. And our second layer learns to combine our bent
planes like this. Now, if we only had two layers, we would just bring these surfaces together to form our final
decision boundary. But we now have a whole additional layer of transformations to apply.
Just as we did in our first layer, we now need to apply our RLU activation function where all of the values on our
surface with heights less than zero are set to zero. In our first layer, this operation folds our planes along linear
fold lines. But now in the second layer of our model, the surfaces we're folding are no longer simple planes. If we add a
plane at Z equals 0 to our first neuron surface, we can see that this surface actually has three separate planes that
all cross Z equals 0. When we apply our RLU activation function and fold up our
surface, we create three separate new fold lines, one for each region that
crosses the Z equals0 plane. And interestingly, these folds are not at the same angle, but actually bend at the
joints of the planes we get from our first layer. So here, a single neuron is able to make three separate folds with
fairly complex geometries. Our second neuron in our second layer applies the same operations, but with different
learned weights, resulting in these three new folds. Now, just as our previous two layers
did, our third and final layer scales and adds our new surfaces together.
After our first layer, the combination of our two RLU folds created four regions for the next layer of our model
to work with. These are easiest to see in a 2D projection like this.
Stacking the new fold lines from our second layer. These new folds at various angles come together in a significantly
more complex tiling of our map with these 10 separate regions. When the
final layer of our network scales and adds together the outputs of our second layer, the resulting surfaces are
composed of the same 10 regions, just with different heights.
The height of these surfaces corresponds to the model's final confidence in our two countries. Bringing these surfaces
together and finding their intersection, we get this final decision boundary, which shows some nice peace-wise linear
curvature around the Belgium regions of our map. So the first layer of our network
creates these two folds and four separate regions on our map which are then split by our second layer into
these 10 regions which are used by our final layer to create these surfaces which intersect in a nice border. The
fact that just adding two additional neurons takes our map from these four regions to these 10 is remarkable to me
especially considering the complex geometry of these 10 regions. If we instead arrange our six neurons in
a two-layer network like this, our model learns to fold four copies of our map like this, resulting in these seven
regions, these surfaces, and this final decision boundary.
This decision boundary isn't necessarily worse than the one learned by our deeper model, but I'm particularly struck by
how much more complex the tiling learned by our deeper model is. Qualitatively,
the tiling of our map learned by our shallow network feels very much like we've just stacked four lines together,
which is exactly what we've done. While the tiling learned by our deeper model feels to me like something entirely
different by repeating our folding, scaling, and combining operations, these operations
are able to compound on themselves, allowing the neurons in our second layer to generate significantly more complex
patterns than they would if they were instead positioned in the first layer of our model.
The compounding analogy is not a coincidence. It turns out that we can show that the maximum number of regions
Exponentially Better?
a rail network like ours can divide our map into grows exponentially with the number of layers in our network. This
equation gives the theoretical maximum number of regions our model can create as a function of the number of neurons
in each layer D, the number of inputs D subi, and the number of layers in our network K, not including our final
output layer. Plugging in d= two neurons per layer, d subi equals 2 inputs, and k
equals 2 layers, we get 2 ^2 * 4 equ= 16 total possible regions for our model.
This is a bit above the 10 regions our model actually learned. If we add another two neuron layer to our model,
our number of regions grows to 2 ^ of 4 * 4 = 64. And adding another layer gets
us to 256 and so on. So each layer theoretically quadruples the number of
regions our model can create in this configuration. This final polomial part of the equation
captures what happens in the final layer of our model. If we cut back down to a
shallow two-layer model K becomes one eliminating the exponential growth term.
As we've seen two-layer networks divide up the input plane by stacking separate RLU folds.
So finding the number of regions we can divide our map into with a two-layer network is equivalent to asking how many
separate regions we can split a plane into with d lines. This is a well-known result in combinatorial geometry with
the answer given by this polomial. So our theory tells us that the maximum
number of regions we can create with a two-layer network grows as a polomial function of our number of neurons while
the number of regions we can create with a deeper network grows exponentially with the number of layers.
Placing 64 neurons in the first layer of a two-layer network like this results in a maximum 281 possible regions while
rearranging these neurons into four layers instead results in a theoretical maximum of over 70 million possible
regions. The difference between these growth rates is compelling and is often pointed
to as a reason for the effectiveness of deep learning. However, these numbers are theoretical upper bounds and as a
number of papers have pointed out, these bounds are very loose. In practice, we typically do not see exponential growth
in the number of regions created by deep networks as we add layers. Let's scale up our own deep network and
see how our number of regions scales with our network and how our fit improves. We left off with this three
layer six neuron model that divided our map into these 10 regions resulting in
this final decision boundary. Let's first expand our model to have eight neurons in each of our first two
layers. The eight folds in our first layer now break up our map into these 19
regions. And the various folds of the surfaces created by our second layer come together in these 102 regions. Our
second layer patterns start to get really interesting. Here the RLU function in our second neuron is folding
our surface along 10 different unique joints. Our final layer scales and combines these outputs into these final
surfaces which intersect like this resulting in this final decision border
capturing the two largest sections of our town nicely. Note that a couple of the neurons in our
second layer don't have any colored regions. This means that the entire surface from our first layer was below Z
equals 0 and all inputs are set to zero by our RLU activation function. Dead
neurons like this are common. And a reminder that gradient descent gives no guarantees about efficiently using our
model architecture. Let's add another eight neuron layer to our model resulting in four total
layers. We can now really start to see the compounding effects of our repeated folding, scaling, and combining
operations. The relu folding happening in this third neuron of our third layer creates these
tiny regions around the border. It's so interesting to me that our model guided by back propagation figures out how to
create all these extra little polygons around our town's borders to capture their detailed structure.
Now, at this scale, it becomes tough to make sense of everything that's happening in 3D space like this. Let's
focus on the regions formed on our 2D map by each layer, the final 3D surfaces, and our final decision
boundary. Let's watch our model learn from this perspective. Before training, here's how our model initially divides
up our input space, creating this decision boundary. Before we start the training process, let's add one more
panel that will track the model's loss as it learns. In less than a 100 gradient descent
steps, our model is able to pick out the core structure of our town
and then is able to progressively tighten its borders as it learns, creating more and more regions around
the fine details of the border. Finally, let's add one more layer,
bringing our total number of layers to five, and increase our width one last time to 32 neurons. Our additional layer
gives us one more tiling of our map. And at this scale, our 3D plot becomes a bit too chaotic to make sense of. So, we'll
just watch the 2D plots in this final training animation. Unlike our smaller models, this deeper model really
benefits from more training steps. Using the extra steps to refine the details of our town's border.
The fact that just four layers with 32 neurons each can learn this level of complexity is remarkable to me. Our
final decision boundary impressively captures every region of our town. It's
incredible to me that a bunch of little linear models can come together to do something so complex and that we can
actually find these solutions using gradient descent.
Neural Networks Demystifed
Around 10 years ago, I released the very first Welch Labs video. It's called Neural Networks Demystified.
Like the series you're watching now, neural networks demystified was a series about how neural networks learn,
focusing on back propagation and gradient descent. Sitting down to work on this series 10
years later, I honestly didn't know where to start. Although most of the core approaches in my old videos are
unchanged, these core ideas have been scaled to solve unbelievably complex problems. And this shocking ability to
scale has led the research community to dig deeper into what makes these models tick. We've learned a great deal in 10
years, but many mysteries remain. In part one of this series, we dug into
lost landscapes, and we saw how the standard mental picture of gradient descent that I presented 10 years ago
really doesn't hold up in the incredibly highdimensional spaces these models operate in. In part two, we dug into the
core mechanics of how models learn, dissecting back propagation in the context of a modern large language
model. Finally, in this video, we saw how deep models are able to recursively fold, scale, and combine their input
spaces, learning incredibly complex patterns with remarkably few neurons.
Maybe in another 10 years, I can make another series like this. We'll have to wait and see what these models can do
then, and how much sense we'll be able to make of how they do it.
The Time I Quit YouTube
Back in 2019, I completely quit Welch Labs. I had just tried going full-time
creating videos, but I wasn't able to earn enough money to make it work. I got frustrated and I quit. I went off and
worked as a machine learning engineer, which was great, but I couldn't shake the feeling that I was really supposed to be making videos. Starting in 2022, I
slowly eased back on Tik Tok and was able to gradually build enough momentum to take another crack at going full-time
last year. When I quit in 2019, I had some time to really think about what kept pulling me back into making videos.
And I realized that deep down it was really about education. I loved math and science as a kid, but I
really disliked the way I had to learn it in school. After undergrad, I really found myself questioning if I even liked
math at all. Only through my own work and study did I fall back into love with math and science years later. And now I
want to use Welsh Labs to make education better. But I've realized for me to be
able to do this, I have to first build a viable business. If I can't support myself and my family, I can't spend the
time I need to make this work. Last year, through sponsorships, poster and book sales, and support on Patreon, I
was able to make about half of what I made as a machine learning engineer. I'm not going to lie, so far, this is a much
harder way to earn a living. My goal this year is to replace my full income.
This will allow me to really reach escape velocity and continue full-time on Welch Labs.
Sponsorships, posters, and book sales are going well this year, but to hit my goal, I need to grow Patreon as well.
Your monthly support on Patreon would mean a lot. As a way to say thank you, and today I'm launching a new reward.
New Patreon Rewards!
