import numpy as np
from scipy.integrate import solve_ivp


class ConditionalDynamic:

    def __init__(self, dynamic = None, event = None, trans = None, nxt = None):
        self.dynamic = dynamic
        self.event = event
        self.trans = trans
        self.nxt = nxt

    def run(self, x0, t_span, t_density = 1000):
        if not self.dynamic:
            raise Exception("Dynamic has not been set!")

        t_eval = np.linspace(t_span[0], t_span[1], np.int((t_span[1]-t_span[0])*t_density))
        if self.event:
            self.event.terminal = True
            self.event.direction = True
            events = [self.event]
        else:
            events = None
        sol = solve_ivp(self.dynamic, t_span, x0, t_eval = t_eval, events = events)
        
        if not sol.success:
            raise Exception(sol.message)
        
        if sol.status == 0:
            return [sol.t, sol.y]
        
        ts = sol.t
        xs = sol.y
        tf = sol.t_events[0][0]
        xf = sol.y_events[0][0]
        if self.trans:
            xf = self.trans(tf, xf)
        
        if self.nxt:
            r = self.nxt.run(xf, [tf, t_span[1]], t_density)
            return [np.hstack([ts, r[0]]), np.hstack([xs, r[1]])]
        
        while True:
            t_eval = np.linspace(tf, t_span[1], np.int((t_span[1]-tf)*t_density))
            sol = solve_ivp(self.dynamic, [tf, t_span[1]], xf, t_eval = t_eval, events = events)
            ts = np.hstack([ts, sol.t])
            xs = np.hstack([xs, sol.y])
            if sol.status == 0:
                break
            tf_new = sol.t_events[0][0]
            xf = sol.y_events[0][0]
            if self.trans:
                xf = self.trans(tf, xf)
            if tf_new-tf < 1e-6:
                break
            tf = tf_new
        return [ts, xs]


