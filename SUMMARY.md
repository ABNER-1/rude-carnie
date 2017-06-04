Rude-Carnie:
  - Two models: 
    - Based on the Levi-Hassner paper.
    - Plugging in the same features into Inception which has been
    trained/finetuned for 2 classes on the Adience dataset (similar to LFW, but
    harder images / more blurry, angles etc.)

    He provides a pretrained model for this - so I'm just using that. There is
    no comparison to Levi-Hassner's model, but I'm assuming its not much
    different as L&H did not seem to mention anything special about their deep
    net layers etc and Inception probably is the state of the art in these
    classification tasks etc.

  - Features:
    - Single look case: Just resize the image, and push in all values into the
    classifier.
    - Multiple looks case: Basically takes a bunch of crops across the image
    (top left, top right etc) and does some manipulations, and then passes all
    of these through the classifier, and averages results.
  
    Single is naturally a lot faster (7-8 times faster than the multiple looks
    case, doing around 500 faces in ~2 mins on halfmoon with cpu). In terms of
    performance, it seems like at least it did not perform any worse than
    multiple (check the esper webpages). In the paper, Levi&Hassner mentioned
    that they used the multiple looks case as an extra step because images in
    Adience had weird angles etc. so if the alignment didn't work, this would
    help, but it wasn't clear if it was required.

  - Results:
    - Softmax layer, gives % probability for M or F. Above 90% seems to be
    pretty good usually - describing that further in the results below.

  - GPU:
    - not immediately clear to me why its not working on the gpu, results are 
    just wrong, probably some tensorflow implementation thing. 
    - The author made a commit 2 days ago trying to fix the gpu stuff, but I
    don't think it works still. Maybe he'll fix it soon.

Below, I describe M / F results on a few runs I did:

VGG Dataset:
  - Labeled dataset, so we can infer gender from the names (using
      gender_guesser). Downloaded Images of celebrities.
  - Passing in cropped images
  - Performs really well (~90%) on a randomly chosen 100 image subset I ran it on. 
  - FIXME: Add numbers on a full run / or at least 1000 faces run.

  - Possible reasons:
      - one difference with the face detections in the videos using scanner was
      that bigger faces - and more prominently, hair often seemed more visible.
      (Might be interesting in general to expand out face detection shots
       slightly so the full head - including the hair - is covered? (see effect
         of hair below))

      - Also these faces are generally solid frontal shots, it appears like the
      gender detector is often wrong with side / other angled shots.

Friends Videos:
  - See the esper page .html file again, no quantitative measures, but the stuff
  below is just how it seemed. Have added the predictions / percentages of
  rude-carnie to the All-Faces tab...(with a horrible UI...ideally would have
  added them to clusters as All faces include faces that were discarded by
  facenet...but had some trouble with getting that working, so for now just
  have this. 0 indicates no face found by facenet)

  - Multiple looks case seems to only do worse(?) or at least not better than
  single looks case. And its much slower, so might as well leave it out.

  - Generally see a lot of mistakes. I guess its still better than average, but
  hard to quantify it. Possible Reasons:
    - Small faces in comparison?
    - Many turned faces / other angles.
    - Usually the face detector seems to be not including the hair,
    particularly for the guys. Might test it by just grabbing a bigger bounding
    box around the faces and see how it does (?)

Fox Obama Video - 10 min shortened clip:
  
  - See the esper .html page file again. 
  - Seems much better than Friends video in general, although not without
  mistakes.
    - Bigger, more frontal images?
  - In general, this may be good enough for us I guess 

TODO: Update results based on tests with the remaining 10 min segments of Obama clips.

Ideas to improve performance:

  - Can do training ourselves, using either inception of Levi&Hassner's models.
  Not sure why this would improve it much though.

  - Can combine it with other tools. In particular, if we had something like
  face tracking in the videos - then we could probably avoid the troubles with
  weird angle / sideways faces as we can get high accuracy for the periods with
  the frontal face.

  - Another option may be to combine it with other features:
    - Hair 
    - Clothes (there seem to be some repos online which match cloth
        similarity/comparison)

  - Would it make sense to perhaps pass in the 128 features from face net into
  inception (instead of just the raw image values) - with the idea that these
 features might encode differences b/w male/female better (?)

Effect of Hair?:
  
  For face recognition:
  - Seems like hair wasn't included in either facenet / or openface. At least
  it isn't mentioned in their papers.
    - This paper seems to conclude hair shouldn't be included. <link>
    - This paper suggests that hair could be useful. Might be interesting to
    have this too, but don't know if the implementation is easily available
    etc.

  For gender recognition:
  - Here hair seems definitely useful information (?) in the majority of the
  cases.. Not sure if rude-carnie models had cropped hair / or at least partially 
  included them in their training set.




