# Roadmap

I want to work on this at least a little bit every day

For now, I would like to break down work into the following stages:

## Data collection

Decide on a set of languages that I want to initially support in inter-language voice transfer

[english, russian, german, french, spanish, portugese, chinese(china), italian, persian, dutch, turkish]

[38, 1, 14, 10, 5, 0.738, 3, 6, 0.884, 0.389]

Find a dataset with a fair number of speakers / utterances per speaker. Probably just going to use mozilla common voice

Once an initial version of this model works well, I'd like to start going through and indexing creative commons videos, and creating datasets with more
speakers. Idea would be to use the current model to split audio up into clips that it thinks are produced by a given speaker, would set a relatively high
confidence threshold. Anything below that, we would hold onto, and use as a validation set to see if we were able to improve the accuracy of the model.
Anything that it was still struggling with, we can manually label, re-train, and repeat. Hopefully once trained on the initial set of languages, we can
perform a similar process for other languages, that aren't very well represented in data sets like Mozilla's open voice. Mozilla has a goal of 10,000
hours, we could hopefully achieve that.

Once we have this encoder trained well, I'd like to use it in a similar method as was used in 
[Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning). It seems like if we used this speaker embedding in a text-to-speech
model, hopefully the model would be able to apply some sorts of "filters" to better detect speech patterns. At least at a surface level,
this seems to make sense to me. Would be cool to see if we could beat state-of-the-art with such a technique. Honestly though, I don't really know
what text-to-speech architectures look like, so I'm not even 100% sure where we would feed in the embedding.

## Data preparation

Write scripts to normalize the data and get it into a format usable by you. This should really be a single 
function that accepts a path to a wav file, and returns the input to the neural net. I think librosa will do this for you

## Model construction

Recreate the model described in the paper

## Training

Train the model on the data. Would like to figure out how to use tensorboard as well, or at least something that would allow me to monitor training remotely

## Inference

Test out the model and see how it performs

## Review

Write some sort of a blog post documenting all of this, and just put it out there. Probably a decent way to start getting feedback. For
the next model, maybe try recording some videos documenting the process.