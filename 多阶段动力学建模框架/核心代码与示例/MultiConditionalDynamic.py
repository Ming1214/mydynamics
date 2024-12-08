import numpy as np
from scipy.integrate import solve_ivp


class MultiConditionalDynamic:

    def __init__(self, dynamic = None, events = None, transes = None, nxts = None):
        self.dynamic = dynamic
        self.events = events
        self.transes = transes
        self.nxts = nxts

    def run(self, x0, t_span, t_density = 1000):
        if not self.dynamic:
            raise Exception("Dynamic has not been set!")

        t_eval = np.linspace(t_span[0], t_span[1], np.int((t_span[1]-t_span[0])*t_density))
        if self.events:
            for event in self.events:
                event.terminal = True
                event.direction = True
            events = self.events
        else:
            events = None
        sol = solve_ivp(self.dynamic, t_span, x0, t_eval = t_eval, events = events)
        
        if not sol.success:
            raise Exception(sol.message)
        
        if sol.status == 0:
            return [sol.t, sol.y]
        
        idx = 0
        for event in self.events:
            if sol.t_events[idx].size > 0:
                break
            idx += 1
        ts = sol.t
        xs = sol.y
        tf = sol.t_events[idx][0]
        xf = sol.y_events[idx][0]
        
        if self.transes:
            xf = self.transes[idx](tf, xf)
        
        if self.nxts:
            r = self.nxts[idx].run(xf, [tf, t_span[1]], t_density)
            return [np.hstack([ts, r[0]]), np.hstack([xs, r[1]])]
        
        while True:
            t_eval = np.linspace(tf, t_span[1], np.int((t_span[1]-tf)*t_density))
            sol = solve_ivp(self.dynamic, [tf, t_span[1]], xf, t_eval = t_eval, events = events)
            ts = np.hstack([ts, sol.t])
            xs = np.hstack([xs, sol.y])
            if sol.status == 0:
                break
            idx = 0
            for event in self.events:
                if sol.t_events[idx].size > 0:
                    break
                idx += 1
            tf_new = sol.t_events[idx][0]
            xf = sol.y_events[idx][0]
            if self.transes:
                xf = self.transes[idx](tf, xf)
            if tf_new-tf < 1e-6:
                break
            tf = tf_new
        return [ts, xs]


