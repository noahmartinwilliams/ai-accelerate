# ai-accelerate

What is this?
-------------

This an artificial intelligence library which aims to replace pytorch and tensorflow. Unlike those *inferior* libraries, *clearly superior* ai-accelerate uses dependent types in Haskell (the greatest language ever made) so that the tensors can line up with ease, while simultaneously minimizing boiler-plate code, and making simple things easy, while difficult things are still possible. Since it uses the accelerate library it can run on both GPUs and CPUs depending on which library gets imported.

Can it really replace pytorch/tensorflow?
-----------------------------------------

Sadly, not yet. The code is still being developed, however it's happening at a rapid pace (contributions welcome). 

What's wrong with pytorch/tensorflow?
-------------------------------------

While some *inferior* libraries attempt to be useful both to engineers and AI researchers by giving the user tensors in python (*gasp*) that need to be put together with tons of boiler-plate, this library aims to minimize boiler-plate, be more appealing to engineers who just want to train an AI, and maybe even be useful to AI researchers (since the library is open source, AI researchers can add their own code to it, and only need to focus on writing small portions of code that only need to fit a small amount of criteria. What more could you want?). 

Since this library is written in Haskell instead of python it will have the benefit of dependent types that can help align tensors whenever possible while python's lack of a type system leads to either constant error checking or runtime errors (or both). Also I don't like the direction python went in with all of it's excessive use of objects.
