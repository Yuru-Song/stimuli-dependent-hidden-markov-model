## Stimulus-driven hidden Markov model

### Introductions



### Notations

| Variable/parameters | Mathematical expression                                      | Comment                                                      |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Stimulus            | $$\mathbf s_{t}$$                                            | size:  [num_feature, num_time], continuous value             |
| Hidden states       | $$q_t$$                                                      | size: [1, num_time], discrete value, ranges in 1~num_state (might need a $q_0$?) |
| Output              | $$y_t$$                                                      | size: [1, num_time], discrete value, ranges in 1~num_out (might need a $y_0$ ?) |
| Transition filter   | $\mathbf F_{m,n} $                                           | size :[num_state, num_state, num_feature], continuous value, element (m,n): m to n |
| Transition matrix   | $$\mathbf \alpha_{m,n,t} = \frac{\exp(\mathbf{F}_{m,n}\mathbf s_t)}{\sum_l\exp(\mathbf{F}_{m,l}\mathbf s_t)}$$ | Size: [num_state, num_state, num_time], time-varying transition matrix, $\sum_n \alpha_{m,n,t} =1$ |
| Emission filter     | $$\mathbf G_{m,i}$$                                          | Size: [num_state, num_out, num_feature]                      |
| Emission matrix     | $$\eta_{m,i,t} = \frac{\exp(\mathbf{G}_{m,i}\mathbf s_t)}{\sum_j\exp(\mathbf{G}_{m,j}\mathbf s_t)}$$ | Size: [num_state, num_out, num_time], time-varying emission matrix |



### Expectation-maximization

Expected complete log-likelihood (ECLL)
$$
\begin{aligned}\langle L(\boldsymbol{\theta} | \mathbf{y}, \mathbf{q}, \mathbf{S})\rangle_{\hat{p}(\mathbf{q})}=& \sum_{n=1}^{N} \hat{p}\left(q_{0}=n\right) \log \pi_{n}+\sum_{t=1}^{T} \sum_{n=1}^{N} \sum_{m=1}^{N} \hat{p}\left(q_{t-1}=n, q_{t}=m\right) \log \alpha_{n m, t} \\ &+\sum_{t=0}^{T} \sum_{n=1}^{N} \hat{p}\left(q_{t}=n\right) \log \eta_{n y_{t}, t} \;\;\;(1)\end{aligned}
$$
To optimize ECLL, we can seperately optimize 
$$
\sum_{t =1}^T\sum_{m = 1}^{N}\hat p(q_{t -1} = n, q_{t}=m)\log \alpha_{n,m,t}\;\;\;(2)
$$

$$
\sum_{t = 1}^{T}\hat p(q_t = n) \log \eta_{n,y_t,t}\;\;\;(3)
$$

Denote: $$U_{nmt} = \hat{p}\left(q_{t-1}=n, q_{t}=m\right)$$, and $$V_{nt} = \hat{p}\left(q_{t}=n\right)$$, they are computed as follows
$$
U_{nmt} = \hat p(q_{t-1} = n, q_t = m) = p(q_{t-1} = n, q_t = m|\mathbf y,\mathbf \theta ,\mathbf S)
 = \frac{a_{n,t}\alpha_{nmt}\eta_{my_{t}}b_{m,t+1}}{p(\mathbf y|\mathbf\theta,\mathbf S)}\;\;\; (4)\\
$$

$$
V_{nt} = \hat p(q_t = n) = p(q_t = n|\mathbf y,\mathbf \theta, \mathbf S) = \frac{a_{n,t}b_{n,t}}{p(\mathbf y|\mathbf \theta,\mathbf S)}\;\;\; (5)
$$

$$
p(\mathbf y|\mathbf \theta, \mathbf S) = \sum_{n = 1}^N a_{n,T} \;\;\;(6)
$$

For (2), 
$$
\sum_{t=1}^{T} \sum_{m=1}^{N} \hat{p}\left(q_{t-1}=n, q_{t}=m\right) \log \alpha_{n, m,t} 
\\=\sum_{t=1}^{T} \sum_{m=1}^{N}U_{nmt}\log\Big(\frac{\exp(\mathbf F_{n,m}\mathbf s_t)}{\sum_l\exp(\mathbf F_{n,l}\mathbf s_t)}\Big)\\
 = \sum_{t=1}^{T} \sum_{m=1}^{N}U_{nmt}\mathbf F_{n,m}\mathbf s_t - \sum_{t=1}^{T} \sum_{m=1}^{N} U_{nmt}\log(\sum_l\exp(\mathbf F_{n,l}\mathbf s_t))\;\;\;(7)\\
$$
The gradient with regard to $\mathbf F_{n,j}$ is,
$$
\frac{\partial (2)}{\partial \mathbf F_{nj}} = \sum_{t = 1}^T\Big(U_{njt}-(\sum_{m = 1}^NU_{nmt})\alpha_{njt} \Big)\mathbf s_t\\
 = \sum_{t = 1}^T\Big(U_{njt} - (\sum_{m = 1}^N U_{nmt})\frac{\exp(\mathbf{F}_{n,j}\mathbf s_t)}{\sum_l\exp(\mathbf{F}_{n,l}\mathbf s_t)}\Big)\mathbf s_t\;\;\; (8)
$$
The second derivative is,
$$
\frac{\partial^2(2)}{\partial \mathbf F_{nj}^2}=-\sum_{t = 1}^T\big(\sum_{m=1}^NU_{nmt}\big)\frac{\partial \alpha_{njt}}{\partial \mathbf F_{nj}}\mathbf s_t
$$
(In fact you need to compute Hessian matrix ... Stop it...)

Similarly, (3) is
$$
\sum_{t = 1}^{T} V_{nt}\log \eta_{ny_t,t} \\
= \sum_{t = 1}^T V_{nt}\log\Big(\frac{\exp(\mathbf G_{ny_t}\mathbf s_t)}{\sum_{j = 1}^{M}\exp(\mathbf G_{nj}\mathbf s_t)}\Big)\\
 = \sum_{t = 1}^T V_{nt}\mathbf G_{n y_t}\mathbf s_t -\sum_{t = 1}^TV_{nt}\log\Big(\sum_{j=1}^{M} \exp \left(\mathbf{G}_{n j} \mathbf{s}_{t}\right)\Big)
$$
(M is the total number of output choices, i.e. num_out in the previous table)

The gradient with regard to $\mathbf{G}_{ni}$ is,
$$
\frac{\partial (3)}{\partial  \mathbf G_{ni}} = \sum_{t = 1}^TV_{nt}\mathbf \delta_{y_t,i}s_t -\sum_{t = 1}^T V_{nt}\frac{\exp(\mathbf{G}_{ni}\mathbf s_t)}{\sum_{j = 1}^M \exp(\mathbf{G}_{nj}\mathbf s_t)}\mathbf s_t\\
 = \sum_{t = 1}^T(\delta_{y_t,i} - \eta_{ni,t})V_{nt}\mathbf s_t \;\;\;(9)
$$
$$\delta_{y_t,i}$$ is an identity function, i.e. = 0 if $y_t \ne i$,  = 1, if $y_t = i$

