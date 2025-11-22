clear all; %close all;
FONTsize = 14;


path = "D:\\CUDAresults\\Basins_Chen_0001.csv";
idx = csvread(path);  %csv with basins indexes
a = csvread(path + "_1.csv"); %avgPeaks values
b = csvread(path + "_2.csv"); %avgInterval values
c = csvread(path + "_3.csv"); %classification of regimes 

% 1 - x, 2 - y, 3 - z
x_index = 1;
y_index = 2;

x = linspace(a(1,1),a(1,2),length(a(1,:))-1);a(1,:) = [];
y = linspace(a(1,1),a(1,2),length(a(1,:))-1);a(1,:) = [];

a(:,end) = [];
b(1,:) = [];b(1,:) = [];b(:,end) = [];
c(1,:) = [];c(1,:) = [];c(:,end) = [];
idx(1,:) = [];idx(1,:) = [];idx(:,end) = [];

len = length(a(:,1));

flag_NAN_a = 0;
flag_NAN_b = 0;


for i=1:len
    for j=1:len
        if a(i,j) == 999 || a(i,j) == 0
            flag_NAN_a = 1;
            a(i,j) = NaN;
        end
        if b(i,j) == 999 || b(i,j) == 0 
            flag_NAN_b = 1;
            b(i,j) = NaN;
        end

        if b(i,j) == -1
            b(i,j) = NaN;
            flag_NAN_b = 1;
        end
    end
end

A = reshape(a,[],1);
B = reshape(b,[],1);
C = reshape(c,[],1);
labels = reshape(idx,[],1);
X = [A B];

min_a = min(min(a));max_a = max(max(a));
delt_color_a = 0.005*(max_a-min_a);

max_idx = max(max(idx));
min_idx = min(min(idx));
cm = turbo(max_idx-min_idx + 1);
cm1 = turbo(256);

ax = figure ("Name","Basins of attraction");
set(gcf, 'Position', [1 100 750 600])
imagesc(x,y,idx);
set(gca,'YDir','normal');
set(gca,'TickLabelInterpreter','latex','FontSize',18);
xlabel("$x(0)$",'Interpreter','latex','FontSize',22);
ylabel("$y(0)$",'Interpreter','latex','FontSize',22);
cb = colorbar;
set(cb,'TickLabelInterpreter','latex','FontSize',FONTsize);
caxis([min_idx max_idx+0.99]);
if flag_NAN_a || min_idx <=0
    cm(-min_idx +1,:) = [1 1 1];
end
colormap(ax,cm);
cbscaleout = [min_idx max_idx];
ticks = (min_idx:1:max_idx)+0.5;
cb.Ticks = ticks;
cb.TickLabels = min_idx:1:max_idx;


%%
figure ("Name","Basins of attraction, all data")
set(gcf, 'Position', [1 30 1400 700])

ax(1) = subplot(2,3,1);
imagesc(x,y,a);
set(gca,'TickLabelInterpreter','latex','FontSize',FONTsize);
title("$\overline{x_{max}}$",'Interpreter','latex','FontSize',FONTsize+4);
if x_index == 1
    xlabel("$x(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
if x_index == 2
    xlabel("$y(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
if x_index == 3
    xlabel("$z(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
if y_index == 1
    ylabel("$x(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
if y_index == 2
    ylabel("$y(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
if y_index == 3
    ylabel("$z(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
cb = colorbar;
set(cb,'TickLabelInterpreter','latex','FontSize',FONTsize);
set(gca,'YDir','normal');

if flag_NAN_a
    colormap(ax(1),[1 1 1; (cm1)]);
else
    colormap(ax(1),(cm1));
end
caxis([min_a-delt_color_a max_a]);

%%
ax(2) = subplot(2,3,2);

for i=1:len
    for j = 1:len
        if b(i,j) < 0
            b(i,j) = nan;
        end
    end
end

min_b = min(min(b));max_b = max(max(b));
delt_color_b = 0.005*(max_b-min_b);

imagesc(x,y,b);
set(gca,'TickLabelInterpreter','latex','FontSize',FONTsize);
title("$\overline{\Delta t}$",'Interpreter','latex','FontSize',FONTsize);
if x_index == 1
    xlabel("$x(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
if x_index == 2
    xlabel("$y(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
if x_index == 3
    xlabel("$z(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
if y_index == 1
    ylabel("$x(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
if y_index == 2
    ylabel("$y(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
if y_index == 3
    ylabel("$z(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
cb = colorbar;
set(cb,'TickLabelInterpreter','latex','FontSize',FONTsize);
set(gca,'YDir','normal');

if flag_NAN_b
    colormap(ax(2),[1 1 1; (cm1)]);
else
    colormap(ax(2),(cm1));
end
caxis([min_b-delt_color_b max_b]);
%%
ax(3) = subplot(2,3,3);
imagesc(x,y,idx);
set(gca,'YDir','normal');
set(gca,'TickLabelInterpreter','latex','FontSize',FONTsize);
title("$\geq$1 - Osc, $\leq$-1 - FP, 0 - Unb",'Interpreter','latex','FontSize',FONTsize);
if x_index == 1
    xlabel("$x(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
if x_index == 2
    xlabel("$y(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
if x_index == 3
    xlabel("$z(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
if y_index == 1
    ylabel("$x(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
if y_index == 2
    ylabel("$y(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
if y_index == 3
    ylabel("$z(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
cb = colorbar;
set(cb,'TickLabelInterpreter','latex','FontSize',FONTsize);
caxis([min_idx max_idx+0.99]);
ticks = (min_idx:1:max_idx)+0.5;
if flag_NAN_b
    cm(-min_idx +1,:) = [1 1 1];
end
cb.Ticks = ticks;
cb.TickLabels = min_idx:1:max_idx;
colormap(ax(3),cm);
%%
ax(4) = subplot(2,3,4);
scatter(X(:,1),X(:,2),30,labels,"filled");
grid on;
set(gca,'TickLabelInterpreter','latex','FontSize',FONTsize);
colormap(ax(4),cm);
xlabel("$\overline{x_{max}}$",'Interpreter','latex','FontSize',FONTsize+4);
ylabel("$\overline{\Delta t}$",'Interpreter','latex','FontSize',FONTsize+4);
minX = min(X(:,1));maxX = max(X(:,1));deltX = 0.1*(maxX - minX);
minY = min(X(:,2));maxY = max(X(:,2));deltY = 0.1*(maxY - minY);
xlim([minX-deltX-0.25 maxX+deltX]);
ylim([minY-deltY maxY+deltY]);
cb = colorbar;
set(cb,'TickLabelInterpreter','latex','FontSize',FONTsize);
caxis([min_idx max_idx+0.99]);
colormap(ax(4),cm);
cbscaleout = [min_idx max_idx];
ticks = (min_idx:1:max_idx)+0.5;
cb.Ticks = ticks;
cb.TickLabels = min_idx:1:max_idx;

%%
ax(5) = subplot(2,3,5);
for i =1:len
    for j = 1:len
        if c(i,j) == 0
            c(i,j) = 3;
        end
        if c(i,j) == -1
            c(i,j) = 2;
        end
    end
end
imagesc(x,y,c);
set(gca,'YDir','normal');
set(gca,'TickLabelInterpreter','latex','FontSize',FONTsize);
if x_index == 1
    xlabel("$x(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
if x_index == 2
    xlabel("$y(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
if x_index == 3
    xlabel("$z(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
if y_index == 1
    ylabel("$x(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
if y_index == 2
    ylabel("$y(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
if y_index == 3
    ylabel("$z(0)$",'Interpreter','latex','FontSize',FONTsize+4);
end
cb = colorbar;
set(cb,'TickLabelInterpreter','latex','FontSize',FONTsize);
caxis([0.5 3.5]);
cb.Ticks = [1 2 3];
cb.TickLabels = ["Oscillations", "Fixed point", "Unbound"];
colormap(ax(5),[0.8353 0.1686 0.1176;0 0.2235 0.6510;  1 1 1]);

axPos = get(ax(4),'position');
axPos(1) = axPos(1)+ 0.5*axPos(3);
axPos(3) = axPos(3)*0.82;
set(ax(4),'position',axPos);

axPos = get(ax(5),'position');
axPos(1) = axPos(1)+ 0.5*axPos(3);
axPos(3) = axPos(3)*0.82;
set(ax(5),'position',axPos);

