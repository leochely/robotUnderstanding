# robotUnderstanding

## Implementation

### RAD
Joints used are 1 (hip), 4 (head), 8 (left hand), 12 (right hand), 16 (left foot), 20 (right foot).

### Custom Representation
Joints used are 1 (hip), 6 (elbow left), 8 (left hand), 10 (right elbow), 14 (left knee), 20 (right foot).

### Histogram Computation
Histograms are generated using numpy.histogram function. I used N = 8 bins for the d<sub>i</sub> bins and M = 10 bins for the θ_i.
