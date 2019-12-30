from test_gaussian_proc import dumb_gaussian as gaussian, kernels
import numpy as np


class bound():
    def __init__(self, boundaries):
        # boundaries is a list of tuples which defines the permitted region for the
        # optmisation problem.
        # e.g. [(-1,1), (-1,1)] defines a square centred on 0 with side 2
        self.boundaries = boundaries
        for boundary in self.boundaries:
            assert boundary[0] < boundary[1]

    def in_boundary(self, x):
        # given x, checks if the vector is inside the space
        assert len(x) == len(self.boundaries)
        for i, boundary in enumerate(self.boundaries):
            if x[i] < boundary[0] or x[i] > boundary[1]:
                return False
        return True

    def line_plane_intersection(self, x, c, n, k):
        # Given line x+ct and plane n.v = k, find the point of intersection
        # return None if it doesn't exist.
        perp = np.dot(n, c)  # this is the component of c perpendicular to the plane
        if perp == 0.0:
            return None
        t = (k - np.dot(n, x)) / perp
        return x + c * t

    def find_endpoints(self, x, c):
        # Given line x+ct, find the two points where they intersect
        # Assume that c is a unit vector
        # Since we're dealing with random vectors, we don't need to care about edge cases
        # as much -- just retry in those cases
        assert len(x) == len(self.boundaries)
        assert len(c) == len(self.boundaries)
        endpoints = []
        for i, bound in enumerate(self.boundaries):
            n = np.zeros(len(self.boundaries))
            n[i] = 1.0
            for k in bound:
                intersect = self.line_plane_intersection(x, c, n, k)
                if self.in_boundary(intersect):
                    endpoints.append(intersect)
        if len(endpoints) != 2:
            raise FloatingPointError(
                'floating point edge case in random vector -- resample')
        return endpoints

    def random(self):
        sample = np.random.uniform(0.0, 1.0, len(self.boundaries))
        for i, val in enumerate(sample):
            lower = self.boundaries[i][0]
            upper = self.boundaries[i][1]
            sample[i] = lower + val * (upper - lower)
        return sample


class ODSBE():
    def __init__(self, kernel, f, boundaries, **kwargs):
        self.f = f
        self.boundaries = boundaries
        self.f_gaussian = gaussian(kernel, **kwargs)
        self.data = {'inputs': [], 'labels': []}

    def _random_vector_opt(self, x):
        endpoints = None
        for _ in range(5):
            vec = np.random.normal(0, 1, len(x))
            vec = vec / np.linalg.norm(vec)
            try:
                endpoints = self.boundaries.find_endpoints(x, vec)
                break
            except FloatingPointError:
                continue
        if not endpoints:
            raise FloatingPointError()  # TODO: change this to the correct error type, create error hierarchy later

        def constrained_opt(t):
            # constrained linear problem. Solve for t in [0,1]
            input = (1 - t) * endpoints[0] + t * endpoints[1]
            return self.f_gaussian.evaluate(input)

        label, arg = self._dumb_1d_opt(constrained_opt, 100)
        return (1 - arg) * endpoints[0] + arg * endpoints[1]

    def _dumb_1d_opt(self, f, h):
        # TODO: This is incredibly crude. replace it ASAP
        arg, label = 0, f(0)
        for x in np.linspace(0, 1, h):
            new_label = f(x)
            if new_label > label:
                label = new_label
                arg = x
        return label, arg

    def _add_data(self, label, input):
        self.f_gaussian.add_single_data(label, input)
        self.data['inputs'].append(input)
        self.data['labels'].append(label)

    def opt(self, iter):
        if not self.data['inputs']:
            new_input = self.boundaries.random()
            self._add_data(self.f(new_input), new_input)
        for _ in range(iter):
            new_input = self._random_vector_opt(self.data['inputs'][-1])
            self._add_data(self.f(new_input), new_input)
        return self.data['labels'][-1],self.data['inputs'][-1]

if __name__ == '__main__':
    b = bound([(-1, 1), (-1, 1)])
    func = lambda x: -(np.linalg.norm(x)**2)
    optimser = ODSBE(
        kernels.SQUARE_EXP,
        func,
        b,
        length = 1
    )
    print(optimser.opt(10))



