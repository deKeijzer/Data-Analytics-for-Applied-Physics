sample = importdata('call_2.csv'); % column 2 3 4 zijn xyz acc

time = sample(:,1);
time = time - time(1,1); % start waarde van de tijd afhalen
time = time/(60^2); % omzetten naar uren
time = time/24; % omzetten naar dagen

acc_x = sample(:,2);
acc_y = sample(:,3); 
acc_z = sample(:,4);

z = sin(acc_x)+cos(acc_y);

plot(time, acc_z, '.')

xlabel('Tijd [dagen]', 'Interpreter', 'latex')
ylabel('Versnelling in $g$ [m/s$^2$]', 'Interpreter', 'latex')
grid on

