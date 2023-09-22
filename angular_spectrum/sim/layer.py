
class Layer:
    def __init__(self, ot, pixel, z, **kwargs):
        '''
        z: distance from the LAST layer 
            if first layer, then the distance from the objective
        '''
        self.X = ot
        self.z = z
        self.pixel = pixel
    
    def phase_modulate(self, X, pixel_x, wl):
        assert (pixel_x - self.pixel).abs() < 1e-10, 'not only support same \
            resolution between image and layer.'
        raise NotImplementedError