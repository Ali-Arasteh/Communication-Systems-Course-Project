A = imread('pictures\2.gif');
figure();
imshow(A);
B = imresize(A,0.125);
figure();
imshow(B);
XLength = 64;
YLength = 64;
SymbolsLength = 256;
Probability = zeros(2,SymbolsLength);
for i=1:SymbolsLength
    Probability(1,i) = i-1;
    Probability(2,i) = length(find(B==(i-1)));
end
for i=1:SymbolsLength
   for j=1:SymbolsLength-i
       if(Probability(2,j) > Probability(2,j+1))
           temp = Probability(:,j);
           Probability(:,j) = Probability(:,j+1);
           Probability(:,j+1) = temp;
       end
   end
end
structs = [];
for i=1:SymbolsLength
    structs = [structs,struct('Code','','Symbol',Probability(1,i),'Probability',Probability(2,i),'LeftNode',[],'RightNode',[])];
end
structs = Huffman(structs);
while (length(structs) > 1)
	structs = Huffman(structs);
end
Tree = structs;
TableSymbol = [];
TableCode = [];
RightNodes = [];
[LeftNode,temp] = Encoder(Tree);
RightNodes = [RightNodes,temp];
flag = 0;
while(~isempty(RightNodes) || flag == 1)
    while(~isempty(LeftNode.LeftNode))
        [LeftNode,temp] = Encoder(LeftNode);
        RightNodes = [RightNodes,temp];
    end
    TableSymbol = [TableSymbol;LeftNode.Symbol];
    TableCode = [TableCode;string(LeftNode.Code)];
    if(isempty(RightNodes))
        break;
    end
    LeftNode = RightNodes(length(RightNodes));
    RightNodes(length(RightNodes)) = []; 
    if(isempty(RightNodes))
       flag = 1; 
    end
end
Symbol = [];
Code = [];
for i=1:SymbolsLength
    Symbol = [Symbol;Probability(1,SymbolsLength-i+1)];
    Code = [Code;TableCode(find(TableSymbol == Symbol(i),1))];
end
Table = table(Symbol,Code);
EncodedString = "";
% Source Encoder
for i=1:YLength
   for j=1:XLength
       EncodedString = EncodedString + Code(find(Symbol == B(i,j)));
   end
end
EncodedCharArray = char(EncodedString);
% parameters difinition
fs = 1e5;
Ts = 1e-2;
fc = 1e4;
fcenter = 1e4;
BW = 5e3;
M = 4;
% Modulation
[ModulatedSignal,fs] = Modulator(fs,Ts,fc,M,EncodedCharArray);
time = 0:1/fs:(length(ModulatedSignal)-1)/fs;
figure();
plot(time,ModulatedSignal)
title('ModulatedSignal in time domain')
xlabel('time')
ylabel('ModulatedSignal')
xlim([0 (length(ModulatedSignal)-1)/fs])
ylim([-sqrt(2/Ts) sqrt(2/Ts)])
% Channel
[FilteredModulatedSignal] = channel(fs,fcenter,BW,ModulatedSignal);
% Demodulation
[DemodulationMatrix] = Demodulator(fs,Ts,fc,FilteredModulatedSignal);
[DecodedString] = Detector(M,DemodulationMatrix);
% Source Decoder
% for using Source Encoder and Source Decoder blocks alone
% DecodedString = EncodedString;
DecodedCharArray = char(DecodedString);
DecodedImage = zeros(YLength,XLength);
flag = 0;
a = 1;
SymbolCode = '';
for i=1:YLength
    for j=1:XLength
       while (flag == 0)
           SymbolCode = DecodedCharArray(1:a);
           if (find(Code == string(SymbolCode)))
               DecodedImage(i,j) = Symbol(find(Code == string(SymbolCode)));
               flag = 1;
           else
               a = a + 1;
           end
       end
       DecodedCharArray(1:a) = [];
       flag = 0;
       a = 1;
       SymbolCode = '';
    end
end
figure();
imshow(uint8(255 * mat2gray(DecodedImage)));
%%
A = imread('pictures\1.gif');
B = imresize(A,0.125);
XLength = 64;
YLength = 64;
SymbolsLength = 256;
Probability = zeros(2,SymbolsLength);
for i=1:SymbolsLength
    Probability(1,i) = i-1;
    Probability(2,i) = length(find(B==(i-1)));
end
for i=1:SymbolsLength
   for j=1:SymbolsLength-i
       if(Probability(2,j) > Probability(2,j+1))
           temp = Probability(:,j);
           Probability(:,j) = Probability(:,j+1);
           Probability(:,j+1) = temp;
       end
   end
end
structs = [];
for i=1:SymbolsLength
    structs = [structs,struct('Code','','Symbol',Probability(1,i),'Probability',Probability(2,i),'LeftNode',[],'RightNode',[])];
end
structs = Huffman(structs);
while (length(structs) > 1)
	structs = Huffman(structs);
end
Tree = structs;
TableSymbol = [];
TableCode = [];
RightNodes = [];
[LeftNode,temp] = Encoder(Tree);
RightNodes = [RightNodes,temp];
flag = 0;
while(~isempty(RightNodes) || flag == 1)
    while(~isempty(LeftNode.LeftNode))
        [LeftNode,temp] = Encoder(LeftNode);
        RightNodes = [RightNodes,temp];
    end
    TableSymbol = [TableSymbol;LeftNode.Symbol];
    TableCode = [TableCode;string(LeftNode.Code)];
    if(isempty(RightNodes))
        break;
    end
    LeftNode = RightNodes(length(RightNodes));
    RightNodes(length(RightNodes)) = []; 
    if(isempty(RightNodes))
       flag = 1; 
    end
end
Symbol = [];
Code = [];
for i=1:SymbolsLength
    Symbol = [Symbol;Probability(1,SymbolsLength-i+1)];
    Code = [Code;TableCode(find(TableSymbol == Symbol(i),1))];
end
Table = table(Symbol,Code);
EncodedString = "";
% Source Encoder
for i=1:YLength
   for j=1:XLength
       EncodedString = EncodedString + Code(find(Symbol == B(i,j)));
   end
end
EncodedCharArray = char(EncodedString);
% parameters difinition
fs = 1e5;
Ts = 1e-2;
fc = 1e4;
fcenter = 1e4;
BW = 5e3;
M = 4;
% Modulation
[ModulatedSignal,fs] = Modulator(fs,Ts,fc,M,EncodedCharArray);
% frequency domain
ModulatedSignalF = 1/fs*fftshift(fft(ModulatedSignal));
frequency = -fs/2:fs/length(ModulatedSignal):fs/2-fs/length(ModulatedSignal);
figure();
plot(frequency,abs(ModulatedSignalF));
title('ModulatedSignal in frequency domain')
xlabel('frequency')
ylabel('ModulatedSignalF')
%%
summation = 0;
for NumberOfPicture=1:40
    A = imread(['pictures\',num2str(NumberOfPicture),'.gif']);
    B = imresize(A,0.125);
    XLength = 64;
    YLength = 64;
    SymbolsLength = 256;
    Probability = zeros(2,SymbolsLength);
    for i=1:SymbolsLength
        Probability(1,i) = i-1;
        Probability(2,i) = length(find(B==(i-1)));
    end
    for i=1:SymbolsLength
       for j=1:SymbolsLength-i
           if(Probability(2,j) > Probability(2,j+1))
               temp = Probability(:,j);
               Probability(:,j) = Probability(:,j+1);
               Probability(:,j+1) = temp;
           end
       end
    end
    structs = [];
    for i=1:SymbolsLength
        structs = [structs,struct('Code','','Symbol',Probability(1,i),'Probability',Probability(2,i),'LeftNode',[],'RightNode',[])];
    end
    structs = Huffman(structs);
    while (length(structs) > 1)
        structs = Huffman(structs);
    end
    Tree = structs;
    TableSymbol = [];
    TableCode = [];
    RightNodes = [];
    [LeftNode,temp] = Encoder(Tree);
    RightNodes = [RightNodes,temp];
    flag = 0;
    while(~isempty(RightNodes) || flag == 1)
        while(~isempty(LeftNode.LeftNode))
            [LeftNode,temp] = Encoder(LeftNode);
            RightNodes = [RightNodes,temp];
        end
        TableSymbol = [TableSymbol;LeftNode.Symbol];
        TableCode = [TableCode;string(LeftNode.Code)];
        if(isempty(RightNodes))
            break;
        end
        LeftNode = RightNodes(length(RightNodes));
        RightNodes(length(RightNodes)) = []; 
        if(isempty(RightNodes))
           flag = 1; 
        end
    end
    Symbol = [];
    Code = [];
    for i=1:SymbolsLength
        Symbol = [Symbol;Probability(1,SymbolsLength-i+1)];
        Code = [Code;TableCode(find(TableSymbol == Symbol(i),1))];
    end
    Table = table(Symbol,Code);
    EncodedString = "";
    % Source Encoder
    for i=1:YLength
       for j=1:XLength
           EncodedString = EncodedString + Code(find(Symbol == B(i,j)));
       end
    end
    EncodedCharArray = char(EncodedString);
    % parameters difinition
    fs = 1e5;
    Ts = 1e-2;
    fc = 1e4;
    fcenter = 1e4;
    BW = 5e3;
    M = 4;
    % Modulation
    [ModulatedSignal,fs] = Modulator(fs,Ts,fc,M,EncodedCharArray);
    ModulatedSignalF = 1/fs*fftshift(fft(ModulatedSignal));
    ModulatedSignalESD = abs(ModulatedSignalF).^2;
    ModulatedSignalESD = downsample(ModulatedSignalESD,100);
    ModulatedSignalTotalEnergy = sum(ModulatedSignalESD);
    PartOfModulatedSignalEnergy = ModulatedSignalESD(find(ModulatedSignalESD == max(ModulatedSignalESD),1));
    for i = 1:length(ModulatedSignal)/2-find(ModulatedSignalESD == max(ModulatedSignalESD),1)-1
       if(PartOfModulatedSignalEnergy > 0.99/2*ModulatedSignalTotalEnergy)
           break;
       end
       PartOfModulatedSignalEnergy = PartOfModulatedSignalEnergy + ModulatedSignalESD(find(ModulatedSignalESD == max(ModulatedSignalESD),1)+i) + ModulatedSignalESD(find(ModulatedSignalESD == max(ModulatedSignalESD),1)-i);
    end
    summation = summation + i*fs/length(ModulatedSignalESD);
end
AverageBW = summation/40;
%%
A = imread('pictures\1.gif');
B = imresize(A,0.125);
XLength = 64;
YLength = 64;
SymbolsLength = 256;
Probability = zeros(2,SymbolsLength);
for i=1:SymbolsLength
    Probability(1,i) = i-1;
    Probability(2,i) = length(find(B==(i-1)));
end
for i=1:SymbolsLength
   for j=1:SymbolsLength-i
       if(Probability(2,j) > Probability(2,j+1))
           temp = Probability(:,j);
           Probability(:,j) = Probability(:,j+1);
           Probability(:,j+1) = temp;
       end
   end
end
structs = [];
for i=1:SymbolsLength
    structs = [structs,struct('Code','','Symbol',Probability(1,i),'Probability',Probability(2,i),'LeftNode',[],'RightNode',[])];
end
structs = Huffman(structs);
while (length(structs) > 1)
	structs = Huffman(structs);
end
Tree = structs;
TableSymbol = [];
TableCode = [];
RightNodes = [];
[LeftNode,temp] = Encoder(Tree);
RightNodes = [RightNodes,temp];
flag = 0;
while(~isempty(RightNodes) || flag == 1)
    while(~isempty(LeftNode.LeftNode))
        [LeftNode,temp] = Encoder(LeftNode);
        RightNodes = [RightNodes,temp];
    end
    TableSymbol = [TableSymbol;LeftNode.Symbol];
    TableCode = [TableCode;string(LeftNode.Code)];
    if(isempty(RightNodes))
        break;
    end
    LeftNode = RightNodes(length(RightNodes));
    RightNodes(length(RightNodes)) = []; 
    if(isempty(RightNodes))
       flag = 1; 
    end
end
Symbol = [];
Code = [];
for i=1:SymbolsLength
    Symbol = [Symbol;Probability(1,SymbolsLength-i+1)];
    Code = [Code;TableCode(find(TableSymbol == Symbol(i),1))];
end
Table = table(Symbol,Code);
EncodedString = "";
% Source Encoder
for i=1:YLength
   for j=1:XLength
       EncodedString = EncodedString + Code(find(Symbol == B(i,j)));
   end
end
EncodedCharArray = char(EncodedString);
% parameters difinition
fs = 1e5;
Ts = 1e-2;
fc = 1e4;
fcenter = 1e4;
BW = 5e3;
M = 4;
% Modulation
[ModulatedSignal,fs] = Modulator(fs,Ts,fc,M,EncodedCharArray);
interval = floor(logspace(0,4,50));
ErrorProbability = zeros(1,length(interval));
for i=interval
    NoisedModulatedSignal = ModulatedSignal + normrnd(0,i,[1,length(ModulatedSignal)]);
    % Channel
    [FilteredModulatedSignal] = channel(fs,fcenter,BW,NoisedModulatedSignal);
    % Demodulation
    [DemodulationMatrix] = Demodulator(fs,Ts,fc,FilteredModulatedSignal);
    [DecodedString] = Detector(M,DemodulationMatrix);
    DecodedCharArray = char(DecodedString);
    EncodedBitArray = zeros(1,length(EncodedCharArray));
    DecodedBitArray = zeros(1,length(EncodedCharArray));
    for j=1:length(EncodedCharArray)
        EncodedBitArray(j) = bin2dec(EncodedCharArray(j));
        DecodedBitArray(j) = bin2dec(DecodedCharArray(j));
    end
    ErrorProbability(find(interval == i)) = sum(bitxor(EncodedBitArray,DecodedBitArray));
end
ErrorProbability = ErrorProbability/length(EncodedCharArray);
figure();
plot(interval,ErrorProbability);
title('ErrorProbability based on noise variance')
xlabel('noise variance')
ylabel('ErrorProbability')
%%
SNRiSummation = 0;
SNRoSummation = 0;
for NumberOfPicture=1:40
    A = imread(['pictures\',num2str(NumberOfPicture),'.gif']);
    B = imresize(A,0.125);
    XLength = 64;
    YLength = 64;
    SymbolsLength = 256;
    Probability = zeros(2,SymbolsLength);
    for i=1:SymbolsLength
        Probability(1,i) = i-1;
        Probability(2,i) = length(find(B==(i-1)));
    end
    for i=1:SymbolsLength
       for j=1:SymbolsLength-i
           if(Probability(2,j) > Probability(2,j+1))
               temp = Probability(:,j);
               Probability(:,j) = Probability(:,j+1);
               Probability(:,j+1) = temp;
           end
       end
    end
    structs = [];
    for i=1:SymbolsLength
        structs = [structs,struct('Code','','Symbol',Probability(1,i),'Probability',Probability(2,i),'LeftNode',[],'RightNode',[])];
    end
    structs = Huffman(structs);
    while (length(structs) > 1)
        structs = Huffman(structs);
    end
    Tree = structs;
    TableSymbol = [];
    TableCode = [];
    RightNodes = [];
    [LeftNode,temp] = Encoder(Tree);
    RightNodes = [RightNodes,temp];
    flag = 0;
    while(~isempty(RightNodes) || flag == 1)
        while(~isempty(LeftNode.LeftNode))
            [LeftNode,temp] = Encoder(LeftNode);
            RightNodes = [RightNodes,temp];
        end
        TableSymbol = [TableSymbol;LeftNode.Symbol];
        TableCode = [TableCode;string(LeftNode.Code)];
        if(isempty(RightNodes))
            break;
        end
        LeftNode = RightNodes(length(RightNodes));
        RightNodes(length(RightNodes)) = []; 
        if(isempty(RightNodes))
           flag = 1; 
        end
    end
    Symbol = [];
    Code = [];
    for i=1:SymbolsLength
        Symbol = [Symbol;Probability(1,SymbolsLength-i+1)];
        Code = [Code;TableCode(find(TableSymbol == Symbol(i),1))];
    end
    Table = table(Symbol,Code);
    EncodedString = "";
    % Source Encoder
    for i=1:YLength
       for j=1:XLength
           EncodedString = EncodedString + Code(find(Symbol == B(i,j)));
       end
    end
    EncodedCharArray = char(EncodedString);
    % parameters difinition
    fs = 1e5;
    Ts = 1e-2;
    fc = 1e4;
    fcenter = 1e4;
    BW = 5e3;
    M = 4;
    % Modulation
    [ModulatedSignal,fs] = Modulator(fs,Ts,fc,M,EncodedCharArray);
    NoiseSignal = normrnd(0,100,[1,length(ModulatedSignal)]);
    % channel
    [FilteredModulatedSignal] = channel(fs,fcenter,BW,ModulatedSignal);
    [FilteredNoiseSignal] = channel(fs,fcenter,BW,NoiseSignal);
    % SNRi
    SPi = FilteredModulatedSignal*FilteredModulatedSignal';
    NPi = FilteredNoiseSignal*FilteredNoiseSignal';
    SNRi = 10*log(SPi/NPi)/log(10);
    SNRiSummation = SNRiSummation + SNRi;
    % Demodulation
    NoisedModulatedSignal = ModulatedSignal + NoiseSignal;
    [FilteredNoisedModulatedSignal] = channel(fs,fcenter,BW,NoisedModulatedSignal);
    [DemodulationMatrix] = Demodulator(fs,Ts,fc,FilteredNoisedModulatedSignal);
    [DecodedString] = Detector(M,DemodulationMatrix);
    DecodedCharArray = char(DecodedString);
    [ModulatedDemodulatedSignal,fs] = Modulator(fs,Ts,fc,M,DecodedCharArray);
    [FilteredModulatedDemodulatedSignal] = channel(fs,fcenter,BW,ModulatedDemodulatedSignal);
    % SNRo
    SPo = SPi;
    NPo = (FilteredModulatedDemodulatedSignal-FilteredModulatedSignal)*(FilteredModulatedDemodulatedSignal-FilteredModulatedSignal)';
    SNRo = 10*log(SPo/NPo)/log(10);
    SNRoSummation = SNRoSummation + SNRo;
end
AverageSNRi = SNRiSummation/40;
AverageSNRo = SNRoSummation/40;
%%
A = imread('pictures\1.gif');
B = imresize(A,0.125);
XLength = 64;
YLength = 64;
SymbolsLength = 256;
Probability = zeros(2,SymbolsLength);
for i=1:SymbolsLength
    Probability(1,i) = i-1;
    Probability(2,i) = length(find(B==(i-1)));
end
for i=1:SymbolsLength
   for j=1:SymbolsLength-i
       if(Probability(2,j) > Probability(2,j+1))
           temp = Probability(:,j);
           Probability(:,j) = Probability(:,j+1);
           Probability(:,j+1) = temp;
       end
   end
end
structs = [];
for i=1:SymbolsLength
    structs = [structs,struct('Code','','Symbol',Probability(1,i),'Probability',Probability(2,i),'LeftNode',[],'RightNode',[])];
end
structs = Huffman(structs);
while (length(structs) > 1)
    structs = Huffman(structs);
end
Tree = structs;
TableSymbol = [];
TableCode = [];
RightNodes = [];
[LeftNode,temp] = Encoder(Tree);
RightNodes = [RightNodes,temp];
flag = 0;
while(~isempty(RightNodes) || flag == 1)
    while(~isempty(LeftNode.LeftNode))
        [LeftNode,temp] = Encoder(LeftNode);
        RightNodes = [RightNodes,temp];
    end
    TableSymbol = [TableSymbol;LeftNode.Symbol];
    TableCode = [TableCode;string(LeftNode.Code)];
    if(isempty(RightNodes))
        break;
    end
    LeftNode = RightNodes(length(RightNodes));
    RightNodes(length(RightNodes)) = []; 
    if(isempty(RightNodes))
       flag = 1; 
    end
end
Symbol = [];
Code = [];
for i=1:SymbolsLength
    Symbol = [Symbol;Probability(1,SymbolsLength-i+1)];
    Code = [Code;TableCode(find(TableSymbol == Symbol(i),1))];
end
Table = table(Symbol,Code);
EncodedString = "";
% Source Encoder
for i=1:YLength
   for j=1:XLength
       EncodedString = EncodedString + Code(find(Symbol == B(i,j)));
   end
end
EncodedCharArray = char(EncodedString);
% parameters difinition
fs = 1e5;
Ts = 1e-2;
fc = 1e4;
fcenter = 1e4;
BW = 5e3;
M = 4;
v = log(M)/log(2);
if (mod(length(EncodedCharArray),v) ~= 0)
    for i=1:v-mod(length(EncodedCharArray),v)
        EncodedCharArray = [EncodedCharArray,'0'];
    end
end
NumberOfSymbols = length(EncodedCharArray)/v;
Color = zeros(length(NumberOfSymbols),1);
for i=1:NumberOfSymbols
    Color(i,1) = 10*bin2dec(EncodedCharArray(i*v-v+1:i*v))+1;
end
% Modulation
[ModulatedSignal,fs] = Modulator(fs,Ts,fc,M,EncodedCharArray);
interval = floor(logspace(0,3,6));
figure();
for i=interval
    NoisedModulatedSignal = ModulatedSignal + normrnd(0,i,[1,length(ModulatedSignal)]);
    % Channel
    [FilteredModulatedSignal] = channel(fs,fcenter,BW,NoisedModulatedSignal);
    % Demodulation
    [DemodulationMatrix] = Demodulator(fs,Ts,fc,FilteredModulatedSignal);
    subplot(2,3,find(interval == i))
    scatter(DemodulationMatrix(:,1),DemodulationMatrix(:,2),1,Color,'filled');
    hold on
    x = linspace(-50,50,1e3);
    plot(x,0*x,'Color','k');
    plot(0*x,x,'Color','k');
    title('Noise variance = ' + string(floor(i)))
    xlim([-5 5])
    ylim([-5 5])
end
%%
A = imread('pictures\1.gif');
B = imresize(A,0.125);
XLength = 64;
YLength = 64;
SymbolsLength = 256;
Probability = zeros(2,SymbolsLength);
for i=1:SymbolsLength
    Probability(1,i) = i-1;
    Probability(2,i) = length(find(B==(i-1)));
end
for i=1:SymbolsLength
   for j=1:SymbolsLength-i
       if(Probability(2,j) > Probability(2,j+1))
           temp = Probability(:,j);
           Probability(:,j) = Probability(:,j+1);
           Probability(:,j+1) = temp;
       end
   end
end
structs = [];
for i=1:SymbolsLength
    structs = [structs,struct('Code','','Symbol',Probability(1,i),'Probability',Probability(2,i),'LeftNode',[],'RightNode',[])];
end
structs = Huffman(structs);
while (length(structs) > 1)
    structs = Huffman(structs);
end
Tree = structs;
TableSymbol = [];
TableCode = [];
RightNodes = [];
[LeftNode,temp] = Encoder(Tree);
RightNodes = [RightNodes,temp];
flag = 0;
while(~isempty(RightNodes) || flag == 1)
    while(~isempty(LeftNode.LeftNode))
        [LeftNode,temp] = Encoder(LeftNode);
        RightNodes = [RightNodes,temp];
    end
    TableSymbol = [TableSymbol;LeftNode.Symbol];
    TableCode = [TableCode;string(LeftNode.Code)];
    if(isempty(RightNodes))
        break;
    end
    LeftNode = RightNodes(length(RightNodes));
    RightNodes(length(RightNodes)) = []; 
    if(isempty(RightNodes))
       flag = 1; 
    end
end
Symbol = [];
Code = [];
for i=1:SymbolsLength
    Symbol = [Symbol;Probability(1,SymbolsLength-i+1)];
    Code = [Code;TableCode(find(TableSymbol == Symbol(i),1))];
end
Table = table(Symbol,Code);
EncodedString = "";
% Source Encoder
for i=1:YLength
   for j=1:XLength
       EncodedString = EncodedString + Code(find(Symbol == B(i,j)));
   end
end
EncodedCharArray = char(EncodedString);
% parameters difinition
fs = 1e5;
Ts = 1e-2;
fc = 1e4;
fcenter = 1e4;
BW = 5e3;
M = 4;
v = log(M)/log(2);
if (mod(length(EncodedCharArray),v) ~= 0)
    for i=1:v-mod(length(EncodedCharArray),v)
        EncodedCharArray = [EncodedCharArray,'0'];
    end
end
NumberOfSymbols = length(EncodedCharArray)/v;
Color = zeros(length(NumberOfSymbols),1);
for i=1:NumberOfSymbols
    Color(i,1) = 10*bin2dec(EncodedCharArray(i*v-v+1:i*v))+1;
end
% Modulation
[ModulatedSignal,fs] = Modulator(fs,Ts,fc,M,EncodedCharArray);
interval = floor(logspace(0,3,6));
for i=[1,2,4]
    figure();
    for j=interval
        NoisedModulatedSignal = ModulatedSignal + normrnd(0,j,[1,length(ModulatedSignal)]);
        % Channel
        [FilteredModulatedSignal] = channel(fs,fcenter,BW,NoisedModulatedSignal);
        % Demodulation
        Phi = i*pi/12;
        [DemodulationMatrix] = DemodulatorWithPhase(fs,Ts,fc,FilteredModulatedSignal,Phi);
        subplot(2,3,find(interval == j))
        scatter(DemodulationMatrix(:,1),DemodulationMatrix(:,2),1,Color,'filled');
        hold on
        x = linspace(-50,50,1e3);
        plot(x,tan(Phi)*x,'Color','k');
        plot(tan(-Phi)*x,x,'Color','k');
        title('Noise variance = ' + string(floor(j)))
        xlim([-5 5])
        ylim([-5 5])
    end 
end
%%
function [Tree] = Huffman(structs)
s = struct('Code','','Symbol','','Probability',structs(1).Probability + structs(2).Probability,'LeftNode',structs(1),'RightNode',structs(2));
structs(1) = [];
structs(1) = s;
i = 1;
while ( i < length(structs) && structs(i).Probability >= structs(i+1).Probability)
    s = structs(i);
    structs(i) = structs(i+1);
    structs(i+1) = s;
    i = i + 1;
end
Tree = structs;
end
%%
function [Left,Right] = Encoder(Tree)
Left = Tree.LeftNode;
Left.Code = Tree.Code;
Left.Code = strcat(Left.Code,'0');
Right = Tree.RightNode;
Right.Code = Tree.Code;
Right.Code = strcat(Right.Code,'1');
end
%%
function [ModulatedSignal,fs] = Modulator(fs,Ts,fc,M,EncodedCharArray)
v = log(M)/log(2);
if (mod(length(EncodedCharArray),v) ~= 0)
    for i=1:v-mod(length(EncodedCharArray),v)
        EncodedCharArray = [EncodedCharArray,'0'];
    end
end
NumberOfSymbols = length(EncodedCharArray)/v;
time = 0:1/fs:Ts-1/fs;
ModulatedSignal = zeros(1,NumberOfSymbols*length(time));
for i=1:NumberOfSymbols
    m = bin2dec(EncodedCharArray(i*v-v+1:i*v));
    ModulatedSignal(i*length(time)-length(time)+1:i*length(time)) = sqrt(2/Ts)*cos(2*pi*fc.*time+(2*m+1)*pi/M);
end
end
%%
function [FilteredModulatedSignal] = channel(fs,fcenter,BW,ModulatedSignal)
FilteredModulatedSignal = bandpass(ModulatedSignal,[fcenter-BW/2 fcenter+BW/2],fs);
end
%%
function [DemodulationMatrix] = Demodulator(fs,Ts,fc,ModulatedSignal)
time = 0:1/fs:Ts-1/fs;
NumberOfMessages = length(ModulatedSignal)/length(time);
DemodulationMatrix = zeros(NumberOfMessages,2);
for i=1:NumberOfMessages
    Scos = sqrt(2/Ts)*cos(2*pi*fc.*time);
    DemodulationMatrix(i,1) = sum(ModulatedSignal(i*length(time)-length(time)+1:i*length(time)).*Scos)/fs;
    Ssin = sqrt(2/Ts)*sin(2*pi*fc.*time);
    DemodulationMatrix(i,2) = -sum(ModulatedSignal(i*length(time)-length(time)+1:i*length(time)).*Ssin)/fs;
end
end
%%
function [DecodedString] = Detector(M,DemodulationMatrix)
v = log(M)/log(2);
DecodedString = "";
Phases = angle(DemodulationMatrix(:,1)+sqrt(-1)*DemodulationMatrix(:,2));
for i=1:length(Phases)
    if (Phases(i) <= 0)
        Phases(i) = Phases(i) + 2*pi;
    end
end
DecodedPhases = floor(Phases*M/(2*pi));
for i=1:length(DecodedPhases)
    DecodedString = strcat(DecodedString,string(dec2bin(DecodedPhases(i),v)));
end
end
%%
function [DemodulationMatrix] = DemodulatorWithPhase(fs,Ts,fc,ModulatedSignal,Phi)
time = 0:1/fs:Ts-1/fs;
NumberOfMessages = length(ModulatedSignal)/length(time);
DemodulationMatrix = zeros(NumberOfMessages,2);
for i=1:NumberOfMessages
    Scos = sqrt(2/Ts)*cos(2*pi*fc.*time-Phi);
    DemodulationMatrix(i,1) = sum(ModulatedSignal(i*length(time)-length(time)+1:i*length(time)).*Scos)/fs;
    Ssin = sqrt(2/Ts)*sin(2*pi*fc.*time-Phi);
    DemodulationMatrix(i,2) = -sum(ModulatedSignal(i*length(time)-length(time)+1:i*length(time)).*Ssin)/fs;
end
end