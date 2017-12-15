read_dir = 'call\';
write_dir = 'call\matlab_data\';
filename = 'call_2.5 Hz.txt';

% strcat() voegt string bij elkaar toe 'x'+'y' = 'xy'

Data = fileread(strcat(read_dir,filename));
Data = strrep(Data, ',', '.');
FID = fopen(strcat(write_dir,filename), 'w');
fwrite(FID, Data, 'char');
fclose(FID);

sample = importdata(strcat(write_dir,filename));
sample = sample.data;
time = sample(:,1); 
voltage = sample(:,2);

% Fourier
fourier_sample = voltage;

Fs = 1000;                  % Sampling frequency                    
T = 1/Fs;                   % Sampling period       
L = size(fourier_sample,1); % Length of signal
t = (0:L-1)*T;              % Time vector

Y = fft(fourier_sample);

P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = Fs*(0:(L/2))/L;
clf

% Subplot 1
subplot(2,1,1); 
plot(f,P1);
xlim([0 5])
grid on
xlabel('f (Hz)')
ylabel('Amplitude [-]')

% [pks,locs] = findpeaks(P1,f, 'MinPeakHeight', 4E-5)
% %text(locs+.02,pks,num2str((1:numel(pks))'))
% text(locs+.02,pks,num2str((1:numel(pks))'))

% [Peak, PeakIdx] = findpeaks(P1, 'MinPeakHeight', 0.8E-4)
% text(f(PeakIdx), Peak, sprintf('Peaks = %.3f', Peak))


% Subplot 2
subplot(2,1,2);
plot(f,P1);
grid on
xlabel('f (Hz)')
ylabel('Amplitude [-]')