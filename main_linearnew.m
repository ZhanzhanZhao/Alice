clc;
clear all
close all
% regrob = [0, 0
% 5.263157894736821, -21.82284980744589
% 13.157894736842081, -15.532734274711856
% 21.05263157894734, -9.242618741976912
% 26.31578947368422, -3.0166880616179697
% 28.94736842105266, 3.14505776636679
% 34.21052631578948, 9.370988446726187
% 39.4736842105263, 15.59691912708604
% 44.73684210526312, 15.725288831835769
% 47.36842105263156, 15.789473684210407
% 52.63157894736844, 22.01540436456935
% 57.89473684210526, 28.2413350449292
% 63.15789473684214, 34.4672657252886
% 68.42105263157896, 34.59563543003833
% 73.68421052631578, 40.821566110397725
% 78.94736842105266, 40.949935815147
% 84.21052631578948, 47.1758664955064
% 89.4736842105263, 47.30423620025613
% 94.73684210526312, 53.530166880615525
% 102.63157894736844, 59.820282413350014
% 110.52631578947376, 60.01283697047484
% 121.05263157894734, 66.36713735558351
% 131.5789473684211, 60.5263157894733
% 139.47368421052641, 72.91399229781746
% 144.73684210526318, 85.23748395378652
% 152.6315789473685, 85.43003851091134
% 160.5263157894737, 79.52503209242559
% 171.05263157894734, 85.87933247753517
% 178.94736842105266, 86.07188703465954
% 184.21052631578954, 98.3953786906286
% 192.10526315789485, 110.7830551989723
% 200.00000000000006, 104.87804878048746
% 202.6315789473685, 92.74711168164276
% 210.5263157894737, 99.0372272143768
% 218.42105263157902, 93.13222079589195];
% 








% robnorm = [0, 0
% 2.427184466019412, 1.6206896551724164
% 0, 1.7586206896551744
% 2.427184466019412, 2
% 2.427184466019412, 2.5517241379310356
% 4.854368932038881, 2.965517241379313
% 4.854368932038881, 2.7586206896551744
% 7.281553398058236, 3.1379310344827616
% 7.281553398058236, 3.2758620689655196
% 14.563106796116529, 3.379310344827587
% 14.563106796116529, 3.4827586206896566
% 19.417475728155352, 3.3448275862068986
% 21.844660194174764, 3.241379310344831
% 29.126213592233, 3.241379310344831
% 33.98058252427188, 3.2758620689655196
% 33.98058252427188, 3.3448275862068986
% 33.98058252427188, 3.448275862068968
% 38.834951456310705, 3.517241379310347
% 46.11650485436894, 3.517241379310347
% 48.54368932038841, 3.4827586206896566
% 48.54368932038841, 3.448275862068968
% 46.11650485436894, 3.448275862068968
% 50.970873786407765, 3.379310344827587
% 50.970873786407765, 3.31034482758621
% 50.970873786407765, 3.241379310344831
% 58.252427184466, 3.379310344827587
% 58.252427184466, 3.448275862068968
% 60.67961165048547, 3.517241379310347
% 63.10679611650488, 3.4137931034482794
% 65.5339805825243, 3.3448275862068986
% 65.5339805825243, 3.241379310344831
% 72.81553398058253, 3.2758620689655196
% 72.81553398058253, 3.3448275862068986
% 77.66990291262141, 3.448275862068968
% 82.52427184466023, 3.517241379310347
% 84.95145631067965, 3.517241379310347
% 87.378640776699, 3.4137931034482794
% 92.23300970873788, 3.31034482758621
% 97.0873786407767, 3.206896551724139
% 101.94174757281553, 3.2758620689655196
% 106.79611650485435, 3.31034482758621
% 111.65048543689323, 3.3448275862068986
% 116.50485436893206, 3.448275862068968
% 121.35922330097083, 3.517241379310347
% 123.7864077669903, 3.5517241379310374
% 126.2135922330097, 3.4827586206896566
% 126.2135922330097, 3.4137931034482794
% 126.2135922330097, 3.3448275862068986
% 128.64077669902912, 3.241379310344831
% 133.495145631068, 3.241379310344831
% 140.77669902912623, 3.241379310344831
% 143.2038834951456, 3.31034482758621
% 143.2038834951456, 3.3448275862068986
% 145.63106796116506, 3.4137931034482794
% 148.05825242718447, 3.517241379310347
% 152.9126213592233, 3.4827586206896566
% 152.9126213592233, 3.4137931034482794
% 155.3398058252427, 3.3448275862068986
% 155.3398058252427, 3.206896551724139
% 160.19417475728153, 3.241379310344831
% 160.19417475728153, 3.3448275862068986
% 162.621359223301, 3.448275862068968
% 162.621359223301, 3.5517241379310374
% 167.47572815533977, 3.5517241379310374
% 169.90291262135923, 3.517241379310347
% 167.47572815533977, 3.4827586206896566
% 172.3300970873786, 3.448275862068968
% 174.75728155339806, 3.3448275862068986
% 179.61165048543688, 3.241379310344831
% 186.89320388349512, 3.241379310344831
% 186.89320388349512, 3.2758620689655196
% 189.32038834951453, 3.3448275862068986
% 191.747572815534, 3.4137931034482794
% 194.1747572815534, 3.448275862068968
% 194.1747572815534, 3.5517241379310374
% 191.747572815534, 3.655172413793105
% 196.60194174757282, 3.379310344827587
% 199.02912621359224, 3.31034482758621
% 199.02912621359224, 3.206896551724139
% 206.31067961165047, 3.2758620689655196
% 206.31067961165047, 3.379310344827587
% 206.31067961165047, 3.517241379310347
% 213.5922330097087, 3.4827586206896566
% 211.1650485436893, 3.379310344827587
% 216.01941747572818, 3.2758620689655196
% 218.4466019417476, 3.17241379310345];

%%%%%%%%Author: Zhanzhan Zhao
%%%%%%%%Time: 2019/04/20

%%%%%%%%Variables Definition
%%% Terminal Time
T = 200; 
%%%time discretization
Ts = 0.01; 
num_step = (T / Ts) + 1;
%%% Continuous system
A = [1.01 0.01 0; 0.01 1.01 0.01; 0 0.01 1.01];
B = eye(3,3);
C = eye(3,3);
D = zeros(3,3);
% sys = ss(Ac, Bc, Cc, Dc);
% %%% Discrete system
% sysd = c2d(sys,Ts,'zoh');
% A = sysd.A;
% B = sysd.B;
% C = sysd.C; %%%full state feedback
% D = sysd.D;
%%% state dimension
state_dim = size(A,1);
%%% control dimension
control_dim = size(B,2);
phi_dim = state_dim * control_dim; %%length of vec(K)
%%% noise and initial condition
sigma_noise = 0.1;
Cov_noise = sigma_noise^2.*eye(state_dim, state_dim);
sigma_x0 = 0;
mu_x0 = 5;
Sigma = norm(sigma_noise.* ones(state_dim, 1),2);
%%%initial condition
%x0 = ones(state_dim,1);
x0 = normrnd(mu_x0,sigma_x0,[state_dim,1]);
%%%weighting parameters
eta = 10; %%state weighting
beta = 1; %%control weighting another very important factor to accelerate the convergence time

[K_opt,S,e] = dlqr(A,B,eta.* eye(state_dim,state_dim),beta.*eye(control_dim,control_dim),zeros(state_dim, control_dim));
Cov_infiniteopt = dlyap(A - B * K_opt,Cov_noise);

%%%History Storage
real_state = zeros(state_dim, num_step);
real_phi = zeros(control_dim, phi_dim, num_step);
Bphi = zeros(state_dim, phi_dim, num_step-1);
real_control = zeros(state_dim, num_step-1);
imp_previous = zeros(state_dim, num_step-1);
fantasy_state = zeros(state_dim, num_step-1);
fantasy_phi = zeros(control_dim, phi_dim, num_step-1);
fantasy_control = zeros(state_dim, num_step-1);
K = zeros(control_dim, state_dim, num_step-1);
vecK = zeros(phi_dim, num_step-1);
y_dual = zeros(1,num_step-1);
real_loss = zeros(num_step,1);
fantasy_loss = zeros(num_step,1);
real_cost = zeros(num_step,1);
fantasy_cost = zeros(num_step,1);

%%%optimal control data storage
opt_state = zeros(state_dim, num_step);
opt_control = zeros(state_dim, num_step-1);
opt_loss = zeros(num_step,1);
opt_cost = zeros(num_step,1);
try_cost = zeros(num_step,1);
regret = zeros(num_step,1);
regret_try = zeros(num_step,1);
%%%try to see we got a stable policy
try_state = zeros(state_dim, num_step);
try_control = zeros(state_dim, num_step-1);
try_loss = zeros(num_step,1);

%%%tuning constraints
alpha = 0.3; %%%will accelerate the convergence time dominantly
aalpha = 0.75;
r = 0.01;   %%%r will delay the speed of convergence; but sometimes it also stablizes the convergence
gamma = 2;%%%stop criterion
a =10;

%%%plots
nor_realstate = zeros(num_step,1);
nor_optstate = zeros(num_step,1);
nor_trystate = zeros(num_step,1);
step =1;

%%% The outermost forloop is the reality world

    %if step == 1 %%%initial time
        real_state(:,step) = x0; 
        real_phi(:,:,step) = kron(real_state(:,step)', eye(control_dim,control_dim));
        Bphi(:,:,step) = B * real_phi(:,:,step);
        K(:,:,step) = zeros(control_dim, state_dim);
        vecK(:,step) = zeros(phi_dim,1);
        real_control(:,step) = real_phi(:,:,step) * vecK(:,step);
        real_loss(step,1) = eta.* real_state(:,step)' * real_state(:,step) + beta.* real_control(:,step)' * real_control(:,step);
        fantasy_loss(step,1) = real_loss(step,1);
        fantasy_cost(step,1) = fantasy_loss(step,1);
        %%%environment update
        real_state(:,step+1) = A * real_state(:,step) + B * real_control(:,step) + normrnd(0,sigma_noise,[state_dim,1]);
        real_phi(:,:,step+1) = kron(real_state(:,step+1)', eye(control_dim,control_dim));
        Bphi(:,:,step+1) = B * real_phi(:,:,step+1);
        imp_previous(:,step) = real_state(:,step+1) - B * real_control(:,step);
        step =2;
 while step <=50
        
        A = [1.01+0.1*step 0.01 0; 0.01 1.01 0.01; 0 0.01 1.01];
      
        cvx_precision high
        cvx_begin
        variables vK(phi_dim)
        %dual variable y
        fantasy_opt = 0;
        KK = reshape(vK,control_dim,state_dim);
        for i = 1:step-1
            fantasy_opt = fantasy_opt + eta .* (norm(imp_previous(:,i) + Bphi(:,:,i) * vK))^2 + beta .* (norm(B * real_phi(:,:,i) * vK))^2;  
        end
        minimize(fantasy_opt + 0.001 * norm(B*KK - B*K(:,:,step-1),2))
        subject to
        %
        %alpha * norm(real_state(:,step-1)) - norm(imp_previous(:,step-1) + Bphi(:,:,step-1) * vK) <=0
        norm(imp_previous(:,step-1) + Bphi(:,:,step-1) * vK) <= alpha * norm(real_state(:,step-1))
%        KK = reshape(vK,control_dim,state_dim);
% %         if step >= 3
 %         norm(B*KK - B*K(:,:,step-1),2) <= 2 * (alpha) + exp(2) * 100 / exp(step)
% %         end
        cvx_end
        fantasy_cost(step,1) = cvx_optval;
        %y_dual(:,step) = y;
        vecK(:,step) = vK;
        K(:,:,step) = reshape(vK,control_dim,state_dim);
        real_control(:,step) = real_phi(:,:,step) * vecK(:,step);
        %%%environment update
        real_state(:,step+1) = A * real_state(:,step) + B * real_control(:,step) + normrnd(0,sigma_noise,[state_dim,1]);
        real_phi(:,:,step+1) = kron(real_state(:,step+1)', eye(control_dim,control_dim));
        Bphi(:,:,step+1) = B * real_phi(:,:,step+1);
        imp_previous(:,step) = real_state(:,step+1) - B * real_control(:,step);
        if step ~= 1
         real_loss(step,1) = eta.* real_state(:,step)' * real_state(:,step) + beta.* real_control(:,step-1)' * real_control(:,step-1);
        else
         real_loss(step,1) = eta.* x0' * x0;
        end
        step = step + 1;
 end
    
    
num_step = step;
S = zeros(state_dim,state_dim,num_step);
%%%%optimal control cost

% opt_state(:,1) = x0;
% for step = 1:num_step
%      opt_control(:,step) = - K_opt * opt_state(:,step); 
%      opt_state(:,step+1) = A * opt_state(:,step) + B * opt_control(:,step) + normrnd(0,sigma_noise,[state_dim,1]);
%      if step ~= 1
%      opt_loss(step,1) = eta.* opt_state(:,step)' * opt_state(:,step) + beta.* opt_control(:,step-1)' * opt_control(:,step-1);
%      else
%      opt_loss(step,1) = eta.* x0' * x0;
%      end
% end


% %%%%try to see how the last control is behaved like
% K_try = reshape(vK,control_dim,state_dim);
% try_state(:,1) = x0;
% for step = 1:num_step
%      try_control(:,step) = - K_try * try_state(:,step); 
%      try_state(:,step+1) = A * try_state(:,step) + B * try_control(:,step);
%      if step ~= num_step
%      try_loss(step,1) = eta.* try_state(:,step)' * try_state(:,step) + beta.* try_control(:,step)' * try_control(:,step);
%      else
%      try_loss(step,1) = eta.* try_state(:,step)' * try_state(:,step);
%      end
% end


%%%%finite horizon LQR to make a comparison
% try_state(:,1) = x0;
% S(:,:,num_step) = eta.* eye(state_dim,state_dim);
% for i = num_step-1:-1:1
%     %K{i} = inv(R + B'*S{i+1}*B)*B'*S{i+1}*A;
%     %S{i} = Q + K{i}'*R*K{i} + (A-B*K{i})'*S{i+1}*(A-B*K{i}); % First form
%     %S{i} = Q+ A'*S{i+1}*A - A'*S{i+1}*B*inv(R+B'*R*B)*B'*S{i+1}*A;% Second form    
%     
%     S(:,:,i) = eta.* eye(state_dim,state_dim) + A'*S(:,:,i+1)*A - A'*S(:,:,i+1)*B*inv(beta.*eye(control_dim,control_dim)+B'*beta.*eye(control_dim,control_dim)*B)*B'*S(:,:,i+1)*A;
%     
%     %S(:,:,i) =  A'*inv(eye(state_dim,state_dim) + S(:,:,i+1)*B*inv(beta.*eye(control_dim,control_dim))*B')*S(:,:,i+1)*A + eta.* eye(state_dim,state_dim); % Third form
% end
% 
% 
% 
% for step = 1:num_step
%      try_control(:,step) = - inv(beta.*eye(control_dim,control_dim))*B'*S(:,:,step)* try_state(:,step); 
%      try_state(:,step+1) = A * try_state(:,step) + B * try_control(:,step) + normrnd(0,sigma_noise,[state_dim,1]);
%      if step ~= num_step
%      try_loss(step,1) = eta.* try_state(:,step)' * try_state(:,step) + beta.* try_control(:,step)' * try_control(:,step);
%      else
%      try_loss(step,1) = eta.* try_state(:,step)' * try_state(:,step);
%      end
% end
% 
% 
% 




%%%%compute cost
% for i = 1:num_step
%     for j = 1 : i
%      real_cost(i) = real_cost(i) + real_loss(j);
% %      opt_cost(i) = opt_cost(i) + opt_loss(j);
% %      try_cost(i) = try_cost(i) + try_loss(j);
%      regret(i) = real_cost(i) - opt_cost(i);
%      regret_try(i) = real_cost(i) - try_cost(i);
%     end 
% end

% figure (1)
% sstep = 1:num_step-1;
% plot(sstep, real_loss(2:num_step,1),'LineWidth',2);
% hold on
% %plot(sstep, fantasy_loss,'*');
% %hold on
% plot(sstep, opt_loss(2:num_step,1),'LineWidth',2);
% hold on
% % plot(sstep, try_loss(2:num_step,1),'LineWidth',2);
% % hold on
% zz = legend('$f_t(K(t))$','$f_t(K^*)$');
% set(zz,'Interpreter','latex','FontSize',30)
% xx = xlabel('Time Step' );
% set(xx,'Interpreter','latex','FontSize',30)
% yy = ylabel('Loss Function');
% set(yy,'Interpreter','latex','FontSize',30)
% set(gca,'FontSize',20)
% hold on
% plot(sstep, try_loss);
% legend('real loss','fantasy loss','optimal loss','last control loss');

% figure (2)
% sstep = 1:num_step-1;
% plot(sstep, real_cost(2:num_step,1),'LineWidth',2);
% hold on
% %plot(sstep, fantasy_cost,'*');
% %hold on
% plot(sstep, opt_cost(2:num_step,1),'LineWidth',2);
% hold on
% % plot(sstep, try_cost(2:num_step,1),'LineWidth',2);
% % hold on
% zz = legend('$\sum_{i=1}^{T} f_i(K(i))$','$\sum_{i=1}^{T} f_i(K^*)$');
% set(zz,'Interpreter','latex','FontSize',30)
% xx = xlabel('Time Step' );
% set(xx,'Interpreter','latex','FontSize',30)
% yy = ylabel('Cummulative Cost');
% set(yy,'Interpreter','latex','FontSize',30)
% set(gca,'FontSize',20)
% 
% 

% figure (3)
% sstep = 1:num_step-1;
% plot(sstep, abs(regret(2:num_step,1)), 'LineWidth',2);
% hold on
% plot(regrob(:,1), abs(regrob(:,1)),'LineWidth',2);
% % plot(sstep, regret_try(2:num_step,1), 'LineWidth',2);
% % hold on
% xx = xlabel('Time Step','FontSize',30);
% set(xx,'Interpreter','latex','FontSize',30)
% yy = ylabel('Reg$_{T}(K^*)$');
% set(yy,'Interpreter','latex','FontSize',30)
% set(gca, 'YTick', [10.^0 10^1 10^2 10^3 10^4])
% set(gca,'FontSize',20)
% set(gca, 'YScale', 'log')
% zz = legend('Alice','Robust');
% set(zz,'Interpreter','latex','FontSize',30)

for i = 1:num_step
    nor_realstate(i) = norm(real_state(:,i),inf);
%     nor_optstate(i) = norm(opt_state(:,i),inf);
%     nor_trystate(i) = norm(try_state(:,i),inf);
end

figure (4)
sstep = 0:num_step-1;
plot(sstep, nor_realstate(1:num_step,1),'LineWidth',2);
hold on
% plot(sstep, nor_optstate(1:num_step,1),'LineWidth',2)
% hold on
% plot(robnorm(:,1), robnorm(:,2),'LineWidth',2)
% hold on
% zz = legend('$||x_t||_{\infty}$ Alice','$||x_t||_{\infty}$ LQR','$||x_t||_{\infty}$ Robust');
% set(zz,'Interpreter','latex','FontSize',30)
xx = xlabel('Time Step' );
set(xx,'Interpreter','latex','FontSize',30)
yy = ylabel('State Norm Value');
set(yy,'Interpreter','latex','FontSize',30)
set(gca,'FontSize',20)
zz = legend('$||x_t||_{\infty}$ Alice');
set(zz,'Interpreter','latex','FontSize',30)


% plot(t,squeeze(theta_eig),'LineWidth',2);
% xx = xlabel('time (s)' );
% set(xx,'Interpreter','latex','FontSize',30)
% yy = ylabel('eigen-angle');
% set(yy,'Interpreter','latex','FontSize',30)
% set(gca,'FontSize',20)
% ana_stateeig = zeros(state_dim,num_step);
% ana_normstate = zeros(1,num_step);
% ana_ABK = zeros(state_dim,state_dim,num_step);
% ana_ABKeig = zeros(state_dim,num_step);
% for t = 2:num_step-1
% ana_state = zeros(state_dim,state_dim);
% for i = 2:t
%    ana_state = ana_state + real_state(:,i-1) * real_state(:,i-1)';
%    ana_normstate(:,t) = ana_normstate(:,t) + norm(real_state(:,i-1),2)^2;
% end
% ana_stateeig(:,t) = eig(ana_state);
% ana_ABK(:,:,t) = A + B * K(:,:,t);
% ana_ABKeig(:,t) = eig(ana_ABK(:,:,t));
% end


% regret1  = load ('reg1.mat','regret');
% regret2  = load('reg2.mat','regret');
% norstate1 = load('norreal1.mat','nor_realstate');
% norstate2 = load('norstate2.mat','nor_realstate');
% noropt1 = load('noropt1.mat','nor_optstate');
% noropt2 = load('noropt2.mat','nor_optstate');
% 
% regret11 = regret1.regret;
% 
% regret22 = regret2.regret;
% norstate11 = norstate1.nor_realstate;
% norstate22 = norstate2.nor_realstate;
% 
% 
% 
% noropt11 = noropt1.nor_optstate;
% noropt22 = noropt2.nor_optstate;
% 
% 
% 
% 
% 
% 
% 
% for i = 1:218
%     regret(i) = (regret11(i,1) + regret22(i,1))/ 2;
%     nor_realstate(i) = (norstate11(i,1) + norstate22(i,1) ) / 2;
%     nor_optstate(i) = (noropt11(i,1) + noropt22(i,1)) / 2;
%   
% end
% num_step = 218;
% 
% figure (3)
% sstep = 1:num_step-1;
% plot(sstep, abs(regret(2:num_step,1)), 'LineWidth',2);
% hold on
% plot(regrob(:,1), abs(regrob(:,1)),'LineWidth',2);
% % plot(sstep, regret_try(2:num_step,1), 'LineWidth',2);
% % hold on
% xx = xlabel('Time Step','FontSize',30);
% set(xx,'Interpreter','latex','FontSize',30)
% yy = ylabel('Reg$_{T}(K^*)$');
% set(yy,'Interpreter','latex','FontSize',30)
% set(gca, 'YTick', [10.^0 10^1 10^2 10^3 10^4])
% set(gca,'FontSize',20)
% set(gca, 'YScale', 'log')
% zz = legend('Alice','Robust');
% set(zz,'Interpreter','latex','FontSize',30)
% 
% figure (4)
% sstep = 0:num_step-1;
% plot(sstep, nor_realstate(1:num_step,1),'LineWidth',2);
% hold on
% plot(sstep, nor_optstate(1:num_step,1),'LineWidth',2)
% hold on
% plot(robnorm(:,1), robnorm(:,2),'LineWidth',2)
% hold on
% zz = legend('$||x_t||_{\infty}$ Alice','$||x_t||_{\infty}$ LQR','$||x_t||_{\infty}$ Robust');
% set(zz,'Interpreter','latex','FontSize',30)
% xx = xlabel('Time Step' );
% set(xx,'Interpreter','latex','FontSize',30)
% yy = ylabel('State Norm Value');
% set(yy,'Interpreter','latex','FontSize',30)
% set(gca,'FontSize',20)
% 
% 
% 
% 
% 
% 
% 
% 
