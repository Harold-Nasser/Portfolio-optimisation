

%% Données
psi     = [0.0378 0 ; 0 0.0947];
phi     = [0.7146 0 ; 0 0.0353];
omega_0 = [0.0412 0 ; 0 1.3655];
B       = [0.3375 -0.072];
lambda  = 2.14e-5;
n_simul = 50000;
T       = 12;
x0      = 100000;


%% Réplication de : “PanelA, deterministic” et “Panel B, perfect foresight” 

rng(123);
% Générer les n_simul f0
f0  = mvnrnd([0 0], omega_0, n_simul);
% Générer les T epsilons pour chaque simulation
eps = mvnrnd([0 0], psi, n_simul*T)';


% Paramètres nécéssaires à l'optimisation (ne dépend pas des simulations)
%   z = [x0; x1; x2; ...; x11; x12] (dim = T+1 x 1)

%   Construction d'une matrice H tel que H = U * diag_H * U'
%   où U = [-1,  0, ..., 0;
%            1, -1, ..., 0;
%            0,  1, ..., 0;
%                   ...
%            0,  0, ..., 1] (dim = T+1 x T)
%   et diag_H = matrice diagonale contenant Lambda (dim = T x T)
U = diag(ones(T,1), -1);
U = U - diag(ones(T+1,1));
U = U(:,1:T);
diag_H  = eye(T) * lambda;

H = U * diag_H * U';

%   Construction des matrices de contraintes d'égalité
beq          = [x0; 0];
Aeq          = zeros(2, T+1);
Aeq(1,1)     = 1; % x0 = 100 000
Aeq(end,end) = 1; % xT = 0

%   Construction des matrices de contraintes d'inégalité
A = U'; % T contraintes pour que x_t - x_t-1 <= 0
b = zeros(T,1);

% Initialisation des vecteurs pour les résultats
total_perfectForesight    = zeros(1,n_simul);
alpha_deterministicPolicy = zeros(1,n_simul);
tc_deterministicPolicy    = zeros(1,n_simul);
time_vec                  = zeros(1,n_simul);


% Début de la boucle de réplication
for i = 1:n_simul
    tic

    % Simuler les facteurs
    f0_i  = f0(i,:)';
    eps_i = eps(:,1+(i-1)*T:i*T);
    
    f_deterministicPolicy = zeros(length(phi),T); % prédictions des facteurs
    f = [f0_i, zeros(2,T)]; % réalisations des facteurs

    for t = 1:T
        delta_f1 = -0.7146 * f(1, t) + eps_i(1,t);
        delta_f2 = -0.0353 * f(2, t) + eps_i(2,t);
    
        f(1,t+1) = f(1,t) + delta_f1;
        f(2,t+1) = f(2,t) + delta_f2;

        f_deterministicPolicy(:,t) = ((eye(length(phi)) - phi)^t) * f0_i;
    end
    
    % Ajustement des dimensions en vue des optimisations
    % car la somme n'inclue pas x_0
    f_tmp = [zeros(length(phi), 1),f(:,2:end)];
    f_deterministicPolicy_tmp = [zeros(length(phi), 1),f_deterministicPolicy];
    

    % Optimisation : "Deterministic Policy"
    f_quadpgrog1 = (-B * f_deterministicPolicy_tmp)'; 
    [z, ~] = quadprog(H, f_quadpgrog1, A, b, Aeq, beq); 
    
    time_vec(i) = toc;
    alpha_deterministicPolicy(i) = (B * f_tmp) * z;
    tc_deterministicPolicy(i)    = -(z' * H * z)/2;

    
    % Optimisation : Bande supérieur avec "Perfect Foresight"    
    f_quadpgrog2  = (-B * f_tmp)'; 
    [z, fval] = quadprog(H, f_quadpgrog2, A, b, Aeq, beq);

    total_perfectForesight(i) = -fval;
    
end


%% Résultats pour  : "PanelA, Deterministic" et "Panel B, Perfect Foresight" 
mean_alpha_deterministicPolicy = mean(alpha_deterministicPolicy);
mean_tc_deterministicPolicy    = mean(tc_deterministicPolicy);
mean_total_deterministicPolicy = mean_alpha_deterministicPolicy + mean_tc_deterministicPolicy;

se_total_deterministicPolicy = std(alpha_deterministicPolicy + tc_deterministicPolicy) / sqrt(n_simul);
se_alpha_deterministicPolicy = std(alpha_deterministicPolicy) / sqrt(n_simul);
se_tc_deterministicPolicy    = std(tc_deterministicPolicy) / sqrt(n_simul);

bound90_alpha_deterministicPolicy = [mean_alpha_deterministicPolicy - (norminv(0.95) * se_alpha_deterministicPolicy), mean_alpha_deterministicPolicy + (norminv(0.95) * se_alpha_deterministicPolicy)];
bound90_tc_deterministicPolicy    = [mean_tc_deterministicPolicy - (norminv(0.95) * se_tc_deterministicPolicy), mean_tc_deterministicPolicy + (norminv(0.95) * se_tc_deterministicPolicy)];
bound90_total_deterministicPolicy = [mean_total_deterministicPolicy - (norminv(0.95) * se_total_deterministicPolicy), mean_total_deterministicPolicy + (norminv(0.95) * se_total_deterministicPolicy)];


mean_total_perfectForesight = mean(total_perfectForesight);
se_total_perfectForesight   = std(total_perfectForesight) / sqrt(n_simul);

bound90_total_perfectForesight = [mean_total_perfectForesight - (norminv(0.95) * se_total_perfectForesight), mean_total_perfectForesight + (norminv(0.95) * se_total_perfectForesight)];


%% Présentation des résultats
PanelA = array2table({'Mean', mean_alpha_deterministicPolicy/1000, mean_tc_deterministicPolicy/1000, mean_total_deterministicPolicy/1000, 100*(6.46 - (mean_total_deterministicPolicy/1000))/6.46, mean(time_vec); ...
                      'SE', se_alpha_deterministicPolicy/1000, se_tc_deterministicPolicy/1000, se_total_deterministicPolicy/1000, nan, nan; ...
                      '90% L.B.', bound90_alpha_deterministicPolicy(1), bound90_tc_deterministicPolicy(1), bound90_total_deterministicPolicy(1), nan, nan;
                      '90% U.B.', bound90_alpha_deterministicPolicy(2), bound90_tc_deterministicPolicy(2), bound90_total_deterministicPolicy(2), nan, nan}, ...
                      'VariableNames', {'Statistics', 'Alpha ($K)', 'TC ($K)', 'Total ($K)', 'Optimality Gap (%)', 'CPU Time (sec.)'});

PanelB = array2table({'Mean', mean_total_perfectForesight/1000; 'SE', se_total_perfectForesight/1000; ...
                      '90% L.B.', bound90_total_perfectForesight(1); '90% U.B.', bound90_total_perfectForesight(2)}, ...
                      'VariableNames', {'Statistics', 'Total ($K)'});
