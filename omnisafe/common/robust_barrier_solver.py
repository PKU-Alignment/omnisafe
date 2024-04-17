import numpy as np
import torch
from cvxopt import matrix
from cvxopt import solvers
from omnisafe.common.utils import to_tensor, prRed, sort_vertices_cclockwise
from qpth.qp import QPFunction

DYNAMICS_MODE = {'Unicycle': {'n_s': 3, 'n_u': 2},   # state = [x y θ]
                 'SimulatedCars': {'n_s': 10, 'n_u': 1},  # state = [x y θ v ω]
                 'Pvtol': {'n_s': 6, 'n_u': 2},  # state = [x y θ v_x v_y thrust]
                 'Pendulum-v1': {'n_s': 3, 'n_u': 1}
                 }  


class CBFQPLayer:

    def __init__(self, env, device='cpu', gamma_b=20, k_d=3.0, l_p=0.03):
        """Constructor of CBFLayer.

        Parameters
        ----------
        env : gym.env
            Gym environment.
        gamma_b : float, optional
            gamma of control barrier certificate.
        k_d : float, optional
            confidence parameter desired (2.0 corresponds to ~95% for example).
        """

        self.device = torch.device(device)

        self.env = env
        self.u_min, self.u_max = self.get_control_bounds()
        self.gamma_b = gamma_b
        
        self.k_d = k_d
        self.l_p = l_p

        self.action_dim = env.action_space.shape[0]

    def get_safe_action(self, state_batch, action_batch, mean_pred_batch, sigma_batch, modular=False, cbf_info_batch=None): # TODO: 迁移的核心在于此，把它用CBF的方法来改写就好
        """

        Parameters
        ----------
        state_batch : torch.tensor or ndarray
        action_batch : torch.tensor or ndarray
            State batch
        mean_pred_batch : torch.tensor or ndarray
            Mean of disturbance
        sigma_batch : torch.tensor or ndarray
            Standard deviation of disturbance

        Returns
        -------
        final_action_batch : torch.tensor
            Safe actions to take in the environment.
        """

        # batch form if only a single data point is passed
        expand_dims = len(state_batch.shape) == 1
        if expand_dims:
            action_batch = action_batch.unsqueeze(0)
            state_batch = state_batch.unsqueeze(0)
            mean_pred_batch = mean_pred_batch.unsqueeze(0)
            sigma_batch = sigma_batch.unsqueeze(0)
            if cbf_info_batch is not None:
                cbf_info_batch = cbf_info_batch.unsqueeze(0)

        if modular:
            final_action = torch.clamp(action_batch, self.u_min.repeat(action_batch.shape[0], 1), self.u_max.repeat(action_batch.shape[0], 1))
        else:
            Ps, qs, Gs, hs = self.get_cbf_qp_constraints(state_batch, action_batch, mean_pred_batch, sigma_batch, modular=modular, cbf_info_batch=cbf_info_batch)
            
            Ps, qs, Gs, hs = Ps.detach().cpu().numpy(), qs.detach().cpu().numpy(), Gs.detach().cpu().numpy(), hs.detach().cpu().numpy()
            batch_size = Ps.shape[0]
            safe_actions = []
            for i in range(batch_size):
                Ps_m = matrix(np.diag([1., 1e16]), tc='d')
                qs_m = matrix(np.zeros(2))
                Gs_m = matrix(np.float64(Gs[i]), tc='d')
                hs_m = matrix(np.float64(hs[i]), tc='d')
                solvers.options['show_progress'] = False
                sol = solvers.qp(Ps_m, qs_m, Gs_m, hs_m)
                safe_action=torch.as_tensor(sol['x'][0], dtype=torch.float32)
                safe_actions.append(safe_action)
            safe_action_batch = torch.as_tensor(safe_actions, dtype=torch.float32, device=self.device).unsqueeze(-1)
            
            # print(action_batch.shape, safe_action_batch.shape)
            # safe_action_batch = self.solve_qp(Ps, qs, Gs, hs)
            final_action = torch.clamp(action_batch + safe_action_batch, self.u_min.repeat(action_batch.shape[0], 1), self.u_max.repeat(action_batch.shape[0], 1))

        return final_action if not expand_dims else final_action.squeeze(0)

    def solve_qp(self, Ps: torch.Tensor, qs: torch.Tensor, Gs: torch.Tensor, hs: torch.Tensor):
        """Solves:
            minimize_{u,eps} 0.5 * u^T P u + q^T u
                subject to G[u,eps]^T <= h

        Parameters
        ----------
        Ps : torch.Tensor
            (batch_size, n_u+1, n_u+1)
        qs : torch.Tensor
            (batch_size, n_u+1)
        Gs : torch.Tensor
            (batch_size, num_ineq_constraints, n_u+1)
        hs : torch.Tensor
            (batch_size, num_ineq_constraints)
        Returns
        -------
        safe_action_batch : torch.tensor
            The solution of the qp without the last dimension (the slack).
        """

        Ghs = torch.cat((Gs, hs.unsqueeze(2)), -1)
        Ghs_norm = torch.max(torch.abs(Ghs), dim=2, keepdim=True)[0]
        Gs /= Ghs_norm
        hs = hs / Ghs_norm.squeeze(-1)
        sol = self.cbf_layer(Ps, qs, Gs, hs, solver_args={"check_Q_spd": False, "maxIter": 100000, "notImprovedLim": 10, "eps": 1e-4})
        safe_action_batch = sol[:, :self.env.action_space.shape[0]]
        return safe_action_batch

    def cbf_layer(self, Qs, ps, Gs, hs, As=None, bs=None, solver_args=None):
        """

        Parameters
        ----------
        Qs : torch.Tensor
        ps : torch.Tensor
        Gs : torch.Tensor
            shape (batch_size, num_ineq_constraints, num_vars)
        hs : torch.Tensor
            shape (batch_size, num_ineq_constraints)
        As : torch.Tensor, optional
        bs : torch.Tensor, optional
        solver_args : dict, optional

        Returns
        -------
        result : torch.Tensor
            Result of QP
        """

        if solver_args is None:
            solver_args = {}

        if As is None or bs is None:
            As = torch.Tensor().to(self.device).double()
            bs = torch.Tensor().to(self.device).double()

        result = QPFunction(verbose=-1, **solver_args)(Qs.double(), ps.double(), Gs.double(), hs.double(), As, bs).float()
        if torch.any(torch.isnan(result)):
            prRed('QP Failed to solve - result is nan == {}!'.format(torch.any(torch.isnan(result))))
            raise Exception('QP Failed to solve')
        return result

    def get_cbf_qp_constraints(self, state_batch, action_batch, mean_pred_batch, sigma_pred_batch, modular=False, cbf_info_batch=None): # TODO: 解耦合的核心在这里
        """Build up matrices required to solve qp
        
        Program specifically solves:
            minimize_{u,eps} 0.5 * u^T P u + q^T u
                subject to G[u,eps]^T <= h

        Each control barrier certificate is of the form:
            dh/dx^T (f_out + g_out u) >= -gamma^b h_out^3 where out here is an output of the state.

        In the case of SafetyGym_point dynamics:
        state = [x y θ v ω]
        state_d = [v*cos(θ) v*sin(θ) omega ω u^v u^ω]

        Quick Note on batch matrix multiplication for matrices A and B:
            - Batch size should be first dim
            - Everything needs to be 3-dimensional
            - E.g. if B is a vec, i.e. shape (batch_size, vec_length) --> .view(batch_size, vec_length, 1)

        Parameters
        ----------
        state_batch : torch.tensor
            current state (check dynamics.py for details on each dynamics' specifics)
        action_batch : torch.tensor
            Nominal control input.
        mean_pred_batch : torch.tensor
            mean disturbance prediction state, dimensions (n_s, n_u)
        sigma_pred_batch : torch.tensor
            standard deviation in additive disturbance after undergoing the output dynamics.
        gamma_b : float, optional
            CBF parameter for the class-Kappa function

        Returns
        -------
        P : torch.tensor
            Quadratic cost matrix in qp (minimize_{u,eps} 0.5 * u^T P u + q^T u)
        q : torch.tensor
            Linear cost vector in qp (minimize_{u,eps} 0.5 * u^T P u + q^T u)
        G : torch.tensor
            Inequality constraint matrix (G[u,eps] <= h) of size (num_constraints, n_u + 1)
        h : torch.tensor
            Inequality constraint vector (G[u,eps] <= h) of size (num_constraints,)
        """

        assert len(state_batch.shape) == 2 and len(action_batch.shape) == 2 and len(mean_pred_batch.shape) == 2 and len(sigma_pred_batch.shape) == 2, print(state_batch.shape, action_batch.shape, mean_pred_batch.shape, sigma_pred_batch.shape)

        batch_size = state_batch.shape[0]
        gamma_b = self.gamma_b

        # Expand dims
        state_batch = torch.unsqueeze(state_batch, -1).to(self.device)
        action_batch = torch.unsqueeze(action_batch, -1).to(self.device)
        mean_pred_batch = torch.unsqueeze(mean_pred_batch, -1).to(self.device)
        sigma_pred_batch = torch.unsqueeze(sigma_pred_batch, -1).to(self.device)

        if self.env.dynamics_mode == 'Pendulum':
            num_constraints = 8
            n_u = action_batch.shape[1]  # dimension of control inputs
            # Inequality constraints (G[u, eps] <= h)
            G = torch.zeros((batch_size, num_constraints, n_u + 1)).to(self.device)  # the extra variable is for epsilon (to make sure qp is always feasible)
            h = torch.zeros((batch_size, num_constraints)).to(self.device)
            
            h1 = torch.FloatTensor([1, 0.01]).unsqueeze(-1).to(self.device)
            h2 = torch.FloatTensor([1, -0.01]).unsqueeze(-1).to(self.device)
            h3 = torch.FloatTensor([-1, 0.01]).unsqueeze(-1).to(self.device)
            h4 = torch.FloatTensor([-1, -0.01]).unsqueeze(-1).to(self.device)
            action_batch_scaled=(action_batch*15.0).squeeze(-1).to(self.device) # TODO: 写的好看点
            
            theta = state_batch[:,0,:].squeeze(-1)
            theta_dot = state_batch[:,1,:].squeeze(-1)
            f_norm = torch.zeros(batch_size, 2).to(self.device)
            # theta [batch_size, 1]
            f_norm[:, 0] = -3*10/2*torch.sin(theta+torch.pi)*self.env.dt + theta
            f_norm[: ,1] = theta_dot - 3*10/2*torch.sin(theta+torch.pi)
            
            g = torch.tensor([3*self.env.dt**2, 3*self.env.dt]).unsqueeze(0).to(self.device)
            
            f = torch.zeros_like(f_norm).to(self.device)
            f[:, 0] = f_norm[:, 0] + mean_pred_batch[:,0,:].squeeze(-1)
            f[:, 1] = f_norm[:, 1] + mean_pred_batch[:,1,:].squeeze(-1)
            G = torch.tensor(
                [
                    [
                        -torch.matmul(g, h1), 
                        -torch.matmul(g, h2), 
                        -torch.matmul(g, h3), 
                        -torch.matmul(g, h4), 
                        1,
                        -1,
                        g[:, 1],
                        -g[:, 1]
                    ],
                    [
                        -1, 
                        -1, 
                        -1, 
                        -1, 
                        0, 
                        0, 
                        0, 
                        0
                    ]
                ]
            ).transpose(0, 1).repeat(batch_size, 1, 1).to(self.device)
            state_batch_squeeze = state_batch.squeeze(-1)
            sigma_pred_batch_squeeze = sigma_pred_batch.squeeze(-1)

            h = torch.cat(
                [
                    self.gamma_b + torch.matmul(f, h1) + torch.matmul(g, h1) * action_batch_scaled - (1 - self.gamma_b) * torch.matmul(state_batch_squeeze, h1) - self.k_d * torch.abs(torch.matmul(sigma_pred_batch_squeeze, h1)),
                    self.gamma_b + torch.matmul(f, h2) + torch.matmul(g, h2) * action_batch_scaled - (1 - self.gamma_b) * torch.matmul(state_batch_squeeze, h2) - self.k_d * torch.abs(torch.matmul(sigma_pred_batch_squeeze, h2)),
                    self.gamma_b + torch.matmul(f, h3) + torch.matmul(g, h3) * action_batch_scaled - (1 - self.gamma_b) * torch.matmul(state_batch_squeeze, h3) - self.k_d * torch.abs(torch.matmul(sigma_pred_batch_squeeze, h3)),
                    self.gamma_b + torch.matmul(f, h4) + torch.matmul(g, h4) * action_batch_scaled - (1 - self.gamma_b) * torch.matmul(state_batch_squeeze, h4) - self.k_d * torch.abs(torch.matmul(sigma_pred_batch_squeeze, h4)),
                    -action_batch_scaled + 15.0,
                    action_batch_scaled + 15.0,
                    -f[:, 1].unsqueeze(-1) - g[:, 1] * action_batch_scaled + 60.0,
                    f[:, 1].unsqueeze(-1) + g[:, 1] * action_batch_scaled + 60.0
                ],
                dim=1
            ).to(self.device)
            P = torch.diag(torch.tensor([1.e0, 1e16])).repeat(batch_size, 1, 1).to(self.device)
            q = torch.zeros((batch_size, self.action_dim + 1)).to(self.device)
        
        elif self.env.dynamics_mode == 'Unicycle':

            num_cbfs = len(self.env.hazards)
            l_p = self.l_p
            buffer = 0.1

            thetas = state_batch[:, 2, :].squeeze(-1)
            c_thetas = torch.cos(thetas)
            s_thetas = torch.sin(thetas)

            # p(x): lookahead output (batch_size, 2)
            ps = torch.zeros((batch_size, 2)).to(self.device)
            ps[:, 0] = state_batch[:, 0, :].squeeze(-1) + l_p * c_thetas
            ps[:, 1] = state_batch[:, 1, :].squeeze(-1) + l_p * s_thetas

            # p_dot(x) = f_p(x) + g_p(x)u + D_p where f_p(x) = 0,  g_p(x) = RL and D_p is the disturbance

            # f_p(x) = [0,...,0]^T
            f_ps = torch.zeros((batch_size, 2, 1)).to(self.device)

            # g_p(x) = RL where L = diag([1, l_p])
            Rs = torch.zeros((batch_size, 2, 2)).to(self.device)
            Rs[:, 0, 0] = c_thetas
            Rs[:, 0, 1] = -s_thetas
            Rs[:, 1, 0] = s_thetas
            Rs[:, 1, 1] = c_thetas
            Ls = torch.zeros((batch_size, 2, 2)).to(self.device)
            Ls[:, 0, 0] = 1
            Ls[:, 1, 1] = l_p
            g_ps = torch.bmm(Rs, Ls)  # (batch_size, 2, 2)

            # D_p(x) = g_p [0 D_θ]^T + [D_x1 D_x2]^T
            mu_theta_aug = torch.zeros([batch_size, 2, 1]).to(self.device)
            mu_theta_aug[:, 1, :] = mean_pred_batch[:, 2, :]
            mu_ps = torch.bmm(g_ps, mu_theta_aug) + mean_pred_batch[:, :2, :]
            sigma_theta_aug = torch.zeros([batch_size, 2, 1]).to(self.device)
            sigma_theta_aug[:, 1, :] = sigma_pred_batch[:, 2, :]
            sigma_ps = torch.bmm(torch.abs(g_ps), sigma_theta_aug) + sigma_pred_batch[:, :2, :]

            # Build RCBFs
            hs = 1e3 * torch.ones((batch_size, num_cbfs), device=self.device)  # the RCBF itself
            dhdps = torch.zeros((batch_size, num_cbfs, 2), device=self.device)
            hazards = self.env.hazards
            for i in range(len(hazards)):
                if hazards[i]['type'] == 'circle':  # 1/2 * (||ps - x_obs||^2 - r^2)
                    obs_loc = to_tensor(hazards[i]['location'], torch.FloatTensor, self.device)
                    hs[:, i] = 0.5 * (torch.sum((ps - obs_loc)**2, dim=1) - (hazards[i]['radius'] + buffer)**2)
                    dhdps[:, i, :] = (ps - obs_loc)
                elif hazards[i]['type'] == 'polygon':  # max_j(h_j) where h_j = 1/2 * (dist2seg_j)^2
                    vertices = sort_vertices_cclockwise(hazards[i]['vertices'])  # (n_v, 2)
                    segments = np.diff(vertices, axis=0,
                                       append=vertices[[0]])  # (n_v, 2) at row i contains vector from v_i to v_i+1
                    segments = to_tensor(segments, torch.FloatTensor, self.device)
                    vertices = to_tensor(vertices, torch.FloatTensor, self.device)
                    # Get max RBCF TODO: Can be optimized
                    for j in range(segments.shape[0]):
                        # Compute Distances to segment
                        dot_products = torch.matmul(ps - vertices[j:j + 1], segments[j]) / torch.sum(
                            segments[j] ** 2)  # (batch_size,)
                        mask0_ = dot_products < 0  # if <0 closest point on segment is vertex j
                        mask1_ = dot_products > 1  # if >0 closest point on segment is vertex j+1
                        mask_ = torch.logical_and(dot_products >= 0,
                                                  dot_products <= 1)  # Else find distance to line l_{v_j, v_j+1}
                        # Compute Distances
                        dists2seg = torch.zeros((batch_size))
                        if mask0_.sum() > 0:
                            dists2seg[mask0_] = torch.linalg.norm(ps[mask0_] - vertices[[j]], dim=1)
                        if mask1_.sum() > 0:
                            dists2seg[mask1_] = torch.linalg.norm(ps[mask1_] - vertices[[(j + 1) % segments.shape[0]]], dim=1)
                        if mask_.sum() > 0:
                            dists2seg[mask_] = torch.linalg.norm(
                                dot_products[mask_, None] * segments[j].tile((torch.sum(mask_), 1)) + vertices[[j]] -
                            ps[mask_], dim=1)
                        # Compute hs_ for this segment
                        hs_ = 0.5 * ((dists2seg ** 2) + 0.5*buffer)  # (batch_size,)
                        # Compute dhdps TODO: Can be optimized to only compute for indices that need updating
                        dhdps_ = torch.zeros((batch_size, 2))
                        if mask0_.sum() > 0:
                            dhdps_[mask0_] = ps[mask0_] - vertices[[j]]
                        if mask1_.sum() > 0:
                            dhdps_[mask1_] = ps[mask1_] - vertices[[(j + 1) % segments.shape[0]]]
                        if mask_.sum() > 0:
                            normal_vec = torch.tensor([segments[j][1], -segments[j][0]])
                            normal_vec /= torch.linalg.norm(normal_vec)
                            dhdps_[mask_] = (ps[mask_]-vertices[j]).matmul(normal_vec) * normal_vec.view((1,2)).repeat(torch.sum(mask_), 1)  # dot products (batch_size, 1)
                        # Find indices to update (closest segment basically, worst case -> CBF boolean and is a min)
                        idxs_to_update = torch.nonzero(hs[:, i] - hs_ > 0)
                        # Update the actual hs to be used in the constraints
                        if idxs_to_update.shape[0] > 0:
                            hs[idxs_to_update, i] = hs_[idxs_to_update]
                            # Compute dhdhps for those indices
                            dhdps[idxs_to_update, i, :] = dhdps_[idxs_to_update, :]
                else:
                    raise Exception('Only obstacles of type `circle` or `polygon` are supported, got: {}'.format(hazards[i]['type']))

            n_u = action_batch.shape[1]  # dimension of control inputs
            num_constraints = num_cbfs + 2 * n_u  # each cbf is a constraint, and we need to add actuator constraints (n_u of them)

            # Inequality constraints (G[u, eps] <= h)
            G = torch.zeros((batch_size, num_constraints, n_u + 1)).to(self.device)  # the extra variable is for epsilon (to make sure qp is always feasible)
            h = torch.zeros((batch_size, num_constraints)).to(self.device)
            ineq_constraint_counter = 0

            # Add inequality constraints
            G[:, :num_cbfs, :n_u] = -torch.bmm(dhdps, g_ps)  # h1^Tg(x)
            G[:, :num_cbfs, n_u] = -1  # for slack
            h[:, :num_cbfs] = gamma_b * (hs ** 3) + (torch.bmm(dhdps, f_ps + mu_ps) - torch.bmm(torch.abs(dhdps), sigma_ps) + torch.bmm(torch.bmm(dhdps, g_ps), action_batch)).squeeze(-1)
            ineq_constraint_counter += num_cbfs

            # Let's also build the cost matrices, vectors to minimize control effort and penalize slack
            P = torch.diag(torch.tensor([1.e0, 1.e-2, 1e5])).repeat(batch_size, 1, 1).to(self.device)
            q = torch.zeros((batch_size, n_u + 1)).to(self.device)

        # Add Actuator Constraints
        n_u = action_batch.shape[1]  # dimension of control inputs

        for c in range(n_u):

            # u_max >= u_nom + u ---> u <= u_max - u_nom
            if self.u_max is not None:
                G[:, ineq_constraint_counter, c] = 1
                h[:, ineq_constraint_counter] = self.u_max[c] - action_batch[:, c].squeeze(-1)
                ineq_constraint_counter += 1

            # u_min <= u_nom + u ---> -u <= u_min - u_nom
            if self.u_min is not None:
                G[:, ineq_constraint_counter, c] = -1
                h[:, ineq_constraint_counter] = -self.u_min[c] + action_batch[:, c].squeeze(-1)
                ineq_constraint_counter += 1

        return P, q, G, h

    def get_control_bounds(self):
        """

        Returns
        -------
        u_min : torch.tensor
            min control input.
        u_max : torch.tensor
            max control input.
        """

        u_min = torch.tensor(self.env.safe_action_space.low).to(self.device)
        u_max = torch.tensor(self.env.safe_action_space.high).to(self.device)

        return u_min, u_max
    