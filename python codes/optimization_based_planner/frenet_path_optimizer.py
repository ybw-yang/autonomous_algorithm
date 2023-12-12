import math
import numpy as np
import osqp
from scipy import sparse

EPLISON = 1e-3

class PathOptimizer:
    def __init__(self, point_nums):
        # the number of path points
        self.point_nums = point_nums
        self.l_dim = 3
        # the number of optimization variables
        self.variable_nums = self.l_dim * point_nums # l, l', l''
        
        # cost weights
        self.w_l = 0.0
        self.w_dl = 0.0
        self.w_ddl = 0.0
        self.w_dddl = 0.0
        self.w_ref_l = 0.0

        # the interval between two points
        self.step_list = []
        self.step_sqr_list = []

        # the reference l list for optimizer
        self.ref_l_list = []

        # initial and end states
        self.init_state = []
        self.end_state = []

        # state bounds 
        self.l_upper_bound = []
        self.l_lower_bound = []
        self.dl_upper_bound = []
        self.dl_lower_bound = []
        self.ddl_upper_bound = []
        self.ddl_lower_bound = []
        self.dddl_upper_bound = []
        self.dddl_lower_bound = []

        # the solution of optimizer
        self.solution_l = []
        self.solution_dl = []
        self.solution_ddl = []
        self.solution_dddl = []
        self.solution_theta = []
        self.solution_kappa = []
        self.solution_dkappa = []
        
    def SetCostingWeights(self, w_l, w_dl, w_ddl, w_dddl, w_ref_l):
        self.w_l = w_l
        self.w_dl = w_dl
        self.w_ddl = w_ddl
        self.w_dddl = w_dddl
        self.w_ref_l = w_ref_l

    def SetReferenceLList(self, ref_l_list):
        self.ref_l_list = ref_l_list

    def SetStepList(self, step_list):
        self.step_list = step_list
        self.step_sqr_list = [x*x for x in self.step_list]

    def SetLBound(self, upper_bound, lower_bound):
        self.l_lower_bound = lower_bound
        self.l_upper_bound = upper_bound

    def SetDlBound(self, upper_bound, lower_bound):
        self.dl_lower_bound = lower_bound
        self.dl_upper_bound = upper_bound

    def SetDdlBound(self, upper_bound, lower_bound):
        self.ddl_lower_bound = lower_bound
        self.ddl_upper_bound = upper_bound

    def SetDddlBound(self, upper_bound, lower_bound):
        self.dddl_lower_bound = lower_bound
        self.dddl_upper_bound = upper_bound

    def SetInitState(self, init_state):
        self.init_state = init_state

    def SetEndState(self, end_state):
        self.end_state = end_state

    def FormulateMatrixP(self):
        # Construct matrix P for objective function.
        P = np.zeros((self.variable_nums, self.variable_nums))
        for i in range(0, self.point_nums):
            P[i*self.l_dim,   i*self.l_dim  ] = self.w_l        # p
            P[i*self.l_dim+1, i*self.l_dim+1] = self.w_dl       # v
            P[i*self.l_dim+2, i*self.l_dim+2] = self.w_ddl      # a

            P[i*self.l_dim, i*self.l_dim] += self.w_ref_l

        # j
        for i in range(0, self.point_nums-1):
            P[i*self.l_dim+2, i*self.l_dim+2] += self.w_dddl/self.step_sqr_list[i]
            P[(i+1)*self.l_dim+2, (i+1)*self.l_dim+2] += self.w_dddl/self.step_sqr_list[i]
            P[(i+1)*self.l_dim+2, i*self.l_dim+2] += -2*self.w_dddl/self.step_sqr_list[i]        

        # weight * 2.0 for osqp quadratic form: 1/2*x'Px + q'x
        P = sparse.csc_matrix(2*P)
        return P

    def FormulateVectorq(self):
        # Construct vector q for objective function.
        q = np.zeros((self.variable_nums, 1))
        for i in range(0, self.point_nums):
            q[i*self.l_dim] = -self.w_ref_l*(self.l_lower_bound[i]+self.l_upper_bound[i])

        return q

    def FormulateAffineConstraint(self):
        # Construct matrix A and vector l, u for constraints.
        A_init = np.zeros([3, self.variable_nums])
        # initial state constraint
        A_init[0, 0] = 1
        A_init[1, 1] = 1
        A_init[2, 2] = 1
        # A_init[3,-3] = 1
        # A_init[4,-2] = 1
        # A_init[5,-1] = 1
        bound_initial = np.zeros([3, 1])
        bound_initial[0] = self.init_state[0]
        bound_initial[1] = self.init_state[1]
        bound_initial[2] = self.init_state[2]
        # bound_initial[3] = self.end_state[0]
        # bound_initial[4] = self.end_state[1]
        # bound_initial[5] = self.end_state[2]
        lb_initial = bound_initial-EPLISON
        ub_initial = bound_initial+EPLISON
        # continuity constraints: A_continuity_v
        A_continuity_v = np.zeros([self.point_nums-1, self.variable_nums])
        ub_continuity_v = np.zeros([self.point_nums-1, 1])+EPLISON
        lb_continuity_v = np.zeros([self.point_nums-1, 1])-EPLISON
        for i in range(0, self.point_nums-1):
            A_continuity_v[i, i*self.l_dim+0] = 0
            A_continuity_v[i, i*self.l_dim+1] = -1
            A_continuity_v[i, i*self.l_dim+2] = -0.5*self.step_list[i]
            A_continuity_v[i, i*self.l_dim+3] = 0
            A_continuity_v[i, i*self.l_dim+4] = 1
            A_continuity_v[i, i*self.l_dim+5] = -0.5*self.step_list[i]

        # continuity constraints: A_continuity_p
        A_continuity_p = np.zeros([self.point_nums-1, self.variable_nums])
        ub_continuity_p = np.zeros([self.point_nums-1, 1])+EPLISON
        lb_continuity_p = np.zeros([self.point_nums-1, 1])-EPLISON
        for i in range(0, self.point_nums-1):
            A_continuity_p[i, i*self.l_dim+0] = -1
            A_continuity_p[i, i*self.l_dim+1] = -self.step_list[i]
            A_continuity_p[i, i*self.l_dim+2] = -1/3*self.step_sqr_list[i]
            A_continuity_p[i, i*self.l_dim+3] = 1
            A_continuity_p[i, i*self.l_dim+4] = 0
            A_continuity_p[i, i*self.l_dim+5] = -1/6*self.step_sqr_list[i]*self.step_list[i]
        # low bound constraints: A_lb
        print ([self.l_upper_bound[i] - self.l_lower_bound[i] for i in range(len(self.l_upper_bound))])
        A_b_pva = np.zeros([3*self.point_nums, self.variable_nums])
        ub_pva = np.zeros([3*self.point_nums, 1])
        lb_pva = np.zeros([3*self.point_nums, 1])
        # print(len(self.l_lower_bound), len(self.l_upper_bound), self.point_nums)
        for i in range(0, self.point_nums):
            A_b_pva[i*3  , i*self.l_dim  ] = 1  # p
            A_b_pva[i*3+1, i*self.l_dim+1] = 1  # v
            A_b_pva[i*3+2, i*self.l_dim+2] = 1  # a
            lb_pva[i*3  ] = self.l_lower_bound[i]
            lb_pva[i*3+1] = self.dl_lower_bound[i]
            lb_pva[i*3+2] = self.ddl_lower_bound[i]
            ub_pva[i*3  ] = self.l_upper_bound[i]
            ub_pva[i*3+1] = self.dl_upper_bound[i]
            ub_pva[i*3+2] = self.ddl_upper_bound[i]
        A_b_j = np.zeros([self.point_nums-1, self.variable_nums]) # j
        ub_j = np.zeros([self.point_nums-1, 1])
        lb_j = np.zeros([self.point_nums-1, 1])
        for i in range(0, self.point_nums-1):
            A_b_j[i, i    *self.l_dim+2] = -1
            A_b_j[i, (i+1)*self.l_dim+2] =  1
            lb_j[i] = self.dddl_lower_bound[i]*self.step_list[i]
            ub_j[i] = self.dddl_upper_bound[i]*self.step_list[i]

        # combined matrix
        A = np.vstack((A_init, A_continuity_v, A_continuity_p, A_b_pva, A_b_j))
        A = sparse.csc_matrix(A)
        # lb
        lb = np.vstack((lb_initial, lb_continuity_v, lb_continuity_p, lb_pva, lb_j))
        # ub
        ub = np.vstack((ub_initial, ub_continuity_v, ub_continuity_p, ub_pva, ub_j))

        return A, lb, ub

    def Solve(self):
        # 1. Construct QP problem (P, q, A, l, u)
        # refer to sparse.csc_matrix doc:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html
        P = self.FormulateMatrixP()
        q = self.FormulateVectorq()

        A, lb, ub = self.FormulateAffineConstraint()


        # 2. Create an OSQP object and solve 
        # please refer to https://osqp.org/docs/examples/setup-and-solve.html
        prob = osqp.OSQP()

        # Setup workspace and change alpha parameter: 用于控制优化步长
        # prob.setup(P, q, A, lb, ub, alpha=0.5)

        prob.setup(P, q, A, lb, ub, polish=True, eps_abs=EPLISON, eps_rel=EPLISON,
                            eps_prim_inf=EPLISON, eps_dual_inf=EPLISON, verbose=True)

        # setting warmstart for l, l', l''
        var_warm_start = np.zeros(self.variable_nums)
        for i in range(0, self.point_nums):
            var_warm_start[i*self.l_dim] = self.ref_l_list[i]
        prob.warm_start(x=var_warm_start)

        # Solve problem
        res = prob.solve()
        # print(res.x)
        # 3. Extract solution from osqp result
        if res.info.status == 'solved':
            self.solution_l.clear()
            self.solution_dl.clear()
            self.solution_ddl.clear()
            self.solution_dddl.clear()
            for i in range(0, self.point_nums):
                self.solution_l.append(res.x[i*self.l_dim])
                self.solution_dl.append(res.x[i*self.l_dim+1])
                self.solution_ddl.append(res.x[i*self.l_dim+2])
                if(i!=0):
                    self.solution_dddl.append((self.solution_ddl[-1]-self.solution_ddl[-2])/self.step_list[i-1])
            self.solution_dddl.append(self.solution_dddl[-1])
            print(len(self.solution_l))
        else:
            print("problem has no solve!")

    def GetSolution(self):
        return self.solution_l, self.solution_dl, self.solution_ddl, self.solution_dddl

# sparse.csc_matrix doc:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html
# example:
# row = [0, 2, 2, 0, 1, 2, 0, 0, 2]
# col = [0, 0, 1, 2, 2, 2, 0, 1, 0]
# data = [1, 2, 3, 4, 5, 6, 3, 8, 2]
# print(sparse.csc_matrix((data, (row, col)), shape=(3, 3)).toarray())