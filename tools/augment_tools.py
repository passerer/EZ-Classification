class CutOut(object):

    def __init__(self, ratio = 0.1):
        assert ratio<1,"cut area must smaller than given image"
        self.ratio = ratio

    def __call__(self, sample):
        # sample : PIL Image
        image= np.array(sample)

        h, w = image.shape[:2]
        cut_h, cut_w = int(self.ratio*h), int(self.ratio*w)

        top = np.random.randint(0, h - cut_h)
        left = np.random.randint(0, w - cut_w)
        
        cut_area = np.zeros((cut_h,cut_w,3))
        image[top: top + cut_h, left: left + cut_w,:]=cut_area

        return Image.fromarray(image)

class RandomErase(object):
    
    def __init__(self, ratio = 0.4):
        assert ratio<1,"cut area must smaller than given image"
        self.ratio = ratio

    def __call__(self, sample):
        # sample: PIL Image
        image= np.array(sample)

        h, w = image.shape[:2]
        cut_h = np.random.randint(0, int(self.ratio*h))
        cut_w = np.random.randint(0, int(self.ratio*w))

        top = np.random.randint(0, h - cut_h)
        left = np.random.randint(0, w - cut_w)
        
        cut_area = np.random.randint(256, size=(cut_h,cut_w,3))
        image[top: top + cut_h, left: left + cut_w]=cut_area

        return Image.fromarray(image)