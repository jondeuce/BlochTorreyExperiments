function [base, flag] = get_base16(name)
    
    [base, flag] = get_base16_(lower(name));
    
end

function [b, flag] = get_base16_(name)
    
    flag = 0;
    
    switch name
        
        case '3024'
            % Base16 3024
            % Scheme: Jan T. Sott (http://github.com/idleberg)
            
            base00 = '0x090300';
            base01 = '0x3a3432';
            base02 = '0x4a4543';
            base03 = '0x5c5855';
            base04 = '0x807d7c';
            base05 = '0xa5a2a2';
            base06 = '0xd6d5d4';
            base07 = '0xf7f7f7';
            base08 = '0xdb2d20';
            base09 = '0xe8bbd0';
            base0A = '0xfded02';
            base0B = '0x01a252';
            base0C = '0xb5e4f4';
            base0D = '0x01a0e4';
            base0E = '0xa16a94';
            base0F = '0xcdab53';
            
            
        case 'apathy'
            % Base16 Apathy
            % Scheme: Jannik Siebert (https://github.com/janniks)
            
            base00 = '0x031A16';
            base01 = '0x0B342D';
            base02 = '0x184E45';
            base03 = '0x2B685E';
            base04 = '0x5F9C92';
            base05 = '0x81B5AC';
            base06 = '0xA7CEC8';
            base07 = '0xD2E7E4';
            base08 = '0x3E9688';
            base09 = '0x3E7996';
            base0A = '0x3E4C96';
            base0B = '0x883E96';
            base0C = '0x963E4C';
            base0D = '0x96883E';
            base0E = '0x4C963E';
            base0F = '0x3E965B';
            
            
        case 'ashes'
            % Base16 Ashes
            % Scheme: Jannik Siebert (https://github.com/janniks)
            
            base00 = '0x1C2023';
            base01 = '0x393F45';
            base02 = '0x565E65';
            base03 = '0x747C84';
            base04 = '0xADB3BA';
            base05 = '0xC7CCD1';
            base06 = '0xDFE2E5';
            base07 = '0xF3F4F5';
            base08 = '0xC7AE95';
            base09 = '0xC7C795';
            base0A = '0xAEC795';
            base0B = '0x95C7AE';
            base0C = '0x95AEC7';
            base0D = '0xAE95C7';
            base0E = '0xC795AE';
            base0F = '0xC79595';
            
            
        case 'atelierdune'
            % Base16 Atelier Dune
            % Scheme: Bram de Haan (http://atelierbram.github.io/syntax-highlighting/atelier-schemes/dune)
            
            base00 = '0x20201d';
            base01 = '0x292824';
            base02 = '0x6e6b5e';
            base03 = '0x7d7a68';
            base04 = '0x999580';
            base05 = '0xa6a28c';
            base06 = '0xe8e4cf';
            base07 = '0xfefbec';
            base08 = '0xd73737';
            base09 = '0xb65611';
            base0A = '0xcfb017';
            base0B = '0x60ac39';
            base0C = '0x1fad83';
            base0D = '0x6684e1';
            base0E = '0xb854d4';
            base0F = '0xd43552';
            
            
        case 'atelierforest'
            % Base16 Atelier Forest
            % Scheme: Bram de Haan (http://atelierbram.github.io/syntax-highlighting/atelier-schemes/forest)
            
            base00 = '0x1b1918';
            base01 = '0x2c2421';
            base02 = '0x68615e';
            base03 = '0x766e6b';
            base04 = '0x9c9491';
            base05 = '0xa8a19f';
            base06 = '0xe6e2e0';
            base07 = '0xf1efee';
            base08 = '0xf22c40';
            base09 = '0xdf5320';
            base0A = '0xd5911a';
            base0B = '0x5ab738';
            base0C = '0x00ad9c';
            base0D = '0x407ee7';
            base0E = '0x6666ea';
            base0F = '0xc33ff3';
            
            
        case 'atelierheath'
            % Base16 Atelier Heath
            % Scheme: Bram de Haan (http://atelierbram.github.io/syntax-highlighting/atelier-schemes/heath)
            
            base00 = '0x1b181b';
            base01 = '0x292329';
            base02 = '0x695d69';
            base03 = '0x776977';
            base04 = '0x9e8f9e';
            base05 = '0xab9bab';
            base06 = '0xd8cad8';
            base07 = '0xf7f3f7';
            base08 = '0xca402b';
            base09 = '0xa65926';
            base0A = '0xbb8a35';
            base0B = '0x379a37';
            base0C = '0x159393';
            base0D = '0x516aec';
            base0E = '0x7b59c0';
            base0F = '0xcc33cc';
            
            
        case 'atelierlakeside'
            % Base16 Atelier Lakeside
            % Scheme: Bram de Haan (http://atelierbram.github.io/syntax-highlighting/atelier-schemes/lakeside/)
            
            base00 = '0x161b1d';
            base01 = '0x1f292e';
            base02 = '0x516d7b';
            base03 = '0x5a7b8c';
            base04 = '0x7195a8';
            base05 = '0x7ea2b4';
            base06 = '0xc1e4f6';
            base07 = '0xebf8ff';
            base08 = '0xd22d72';
            base09 = '0x935c25';
            base0A = '0x8a8a0f';
            base0B = '0x568c3b';
            base0C = '0x2d8f6f';
            base0D = '0x257fad';
            base0E = '0x5d5db1';
            base0F = '0xb72dd2';
            
            
        case 'atelierseaside'
            % Base16 Atelier Seaside
            % Scheme: Bram de Haan (http://atelierbram.github.io/syntax-highlighting/atelier-schemes/seaside/)
            
            base00 = '0x131513';
            base01 = '0x242924';
            base02 = '0x5e6e5e';
            base03 = '0x687d68';
            base04 = '0x809980';
            base05 = '0x8ca68c';
            base06 = '0xcfe8cf';
            base07 = '0xf0fff0';
            base08 = '0xe6193c';
            base09 = '0x87711d';
            base0A = '0xc3c322';
            base0B = '0x29a329';
            base0C = '0x1999b3';
            base0D = '0x3d62f5';
            base0E = '0xad2bee';
            base0F = '0xe619c3';
            
            
        case 'bespin'
            % Base16 Bespin
            % Scheme: Jan T. Sott
            
            base00 = '0x28211c';
            base01 = '0x36312e';
            base02 = '0x5e5d5c';
            base03 = '0x666666';
            base04 = '0x797977';
            base05 = '0x8a8986';
            base06 = '0x9d9b97';
            base07 = '0xbaae9e';
            base08 = '0xcf6a4c';
            base09 = '0xcf7d34';
            base0A = '0xf9ee98';
            base0B = '0x54be0d';
            base0C = '0xafc4db';
            base0D = '0x5ea6ea';
            base0E = '0x9b859d';
            base0F = '0x937121';
            
            
        case 'brewer'
            % Base16 Brewer
            % Scheme: Timothée Poisot (http://github.com/tpoisot)
            
            base00 = '0x0c0d0e';
            base01 = '0x2e2f30';
            base02 = '0x515253';
            base03 = '0x737475';
            base04 = '0x959697';
            base05 = '0xb7b8b9';
            base06 = '0xdadbdc';
            base07 = '0xfcfdfe';
            base08 = '0xe31a1c';
            base09 = '0xe6550d';
            base0A = '0xdca060';
            base0B = '0x31a354';
            base0C = '0x80b1d3';
            base0D = '0x3182bd';
            base0E = '0x756bb1';
            base0F = '0xb15928';
            
            
        case 'bright'
            % Base16 Bright
            % Scheme: Chris Kempson (http://chriskempson.com)
            
            base00 = '0x000000';
            base01 = '0x303030';
            base02 = '0x505050';
            base03 = '0xb0b0b0';
            base04 = '0xd0d0d0';
            base05 = '0xe0e0e0';
            base06 = '0xf5f5f5';
            base07 = '0xffffff';
            base08 = '0xfb0120';
            base09 = '0xfc6d24';
            base0A = '0xfda331';
            base0B = '0xa1c659';
            base0C = '0x76c7b7';
            base0D = '0x6fb3d2';
            base0E = '0xd381c3';
            base0F = '0xbe643c';
            
            
        case 'chalk'
            % Base16 Chalk
            % Scheme: Chris Kempson (http://chriskempson.com)
            
            base00 = '0x151515';
            base01 = '0x202020';
            base02 = '0x303030';
            base03 = '0x505050';
            base04 = '0xb0b0b0';
            base05 = '0xd0d0d0';
            base06 = '0xe0e0e0';
            base07 = '0xf5f5f5';
            base08 = '0xfb9fb1';
            base09 = '0xeda987';
            base0A = '0xddb26f';
            base0B = '0xacc267';
            base0C = '0x12cfc0';
            base0D = '0x6fc2ef';
            base0E = '0xe1a3ee';
            base0F = '0xdeaf8f';
            
            
        case 'codeschool'
            % Base16 Codeschool
            % Scheme: brettof86
            
            base00 = '0x232c31';
            base01 = '0x1c3657';
            base02 = '0x2a343a';
            base03 = '0x3f4944';
            base04 = '0x84898c';
            base05 = '0x9ea7a6';
            base06 = '0xa7cfa3';
            base07 = '0xb5d8f6';
            base08 = '0x2a5491';
            base09 = '0x43820d';
            base0A = '0xa03b1e';
            base0B = '0x237986';
            base0C = '0xb02f30';
            base0D = '0x484d79';
            base0E = '0xc59820';
            base0F = '0xc98344';
            
            
        case 'colors'
            % Base16 Colors
            % Scheme: mrmrs (http://clrs.cc)
            
            base00 = '0x111111';
            base01 = '0x333333';
            base02 = '0x555555';
            base03 = '0x777777';
            base04 = '0x999999';
            base05 = '0xbbbbbb';
            base06 = '0xdddddd';
            base07 = '0xffffff';
            base08 = '0xff4136';
            base09 = '0xff851b';
            base0A = '0xffdc00';
            base0B = '0x2ecc40';
            base0C = '0x7fdbff';
            base0D = '0x0074d9';
            base0E = '0xb10dc9';
            base0F = '0x85144b';
            
            
        case 'default'
            % Base16 Default
            % Scheme: Chris Kempson (http://chriskempson.com)
            
            base00 = '0x181818';
            base01 = '0x282828';
            base02 = '0x383838';
            base03 = '0x585858';
            base04 = '0xb8b8b8';
            base05 = '0xd8d8d8';
            base06 = '0xe8e8e8';
            base07 = '0xf8f8f8';
            base08 = '0xab4642';
            base09 = '0xdc9656';
            base0A = '0xf7ca88';
            base0B = '0xa1b56c';
            base0C = '0x86c1b9';
            base0D = '0x7cafc2';
            base0E = '0xba8baf';
            base0F = '0xa16946';
            
            
        case 'eighties'
            % Base16 Eighties
            % Scheme: Chris Kempson (http://chriskempson.com)
            
            base00 = '0x2d2d2d';
            base01 = '0x393939';
            base02 = '0x515151';
            base03 = '0x747369';
            base04 = '0xa09f93';
            base05 = '0xd3d0c8';
            base06 = '0xe8e6df';
            base07 = '0xf2f0ec';
            base08 = '0xf2777a';
            base09 = '0xf99157';
            base0A = '0xffcc66';
            base0B = '0x99cc99';
            base0C = '0x66cccc';
            base0D = '0x6699cc';
            base0E = '0xcc99cc';
            base0F = '0xd27b53';
            
            
        case 'embers'
            % Base16 Embers
            % Scheme: Jannik Siebert (https://github.com/janniks)
            
            base00 = '0x16130F';
            base01 = '0x2C2620';
            base02 = '0x433B32';
            base03 = '0x5A5047';
            base04 = '0x8A8075';
            base05 = '0xA39A90';
            base06 = '0xBEB6AE';
            base07 = '0xDBD6D1';
            base08 = '0x826D57';
            base09 = '0x828257';
            base0A = '0x6D8257';
            base0B = '0x57826D';
            base0C = '0x576D82';
            base0D = '0x6D5782';
            base0E = '0x82576D';
            base0F = '0x825757';
            
            
        case 'flat'
            % Base16 Flat
            % Scheme: Chris Kempson (http://chriskempson.com)
            
            base00 = '0x2C3E50';
            base01 = '0x34495E';
            base02 = '0x7F8C8D';
            base03 = '0x95A5A6';
            base04 = '0xBDC3C7';
            base05 = '0xe0e0e0';
            base06 = '0xf5f5f5';
            base07 = '0xECF0F1';
            base08 = '0xE74C3C';
            base09 = '0xE67E22';
            base0A = '0xF1C40F';
            base0B = '0x2ECC71';
            base0C = '0x1ABC9C';
            base0D = '0x3498DB';
            base0E = '0x9B59B6';
            base0F = '0xbe643c';
            
            
        case 'google'
            % Base16 Google
            % Scheme: Seth Wright (http://sethawright.com)
            
            base00 = '0x1d1f21';
            base01 = '0x282a2e';
            base02 = '0x373b41';
            base03 = '0x969896';
            base04 = '0xb4b7b4';
            base05 = '0xc5c8c6';
            base06 = '0xe0e0e0';
            base07 = '0xffffff';
            base08 = '0xCC342B';
            base09 = '0xF96A38';
            base0A = '0xFBA922';
            base0B = '0x198844';
            base0C = '0x3971ED';
            base0D = '0x3971ED';
            base0E = '0xA36AC7';
            base0F = '0x3971ED';
            
            
        case 'grayscale'
            % Base16 Grayscale
            % Scheme: Alexandre Gavioli (https://github.com/Alexx2/)
            
            base00 = '0x101010';
            base01 = '0x252525';
            base02 = '0x464646';
            base03 = '0x525252';
            base04 = '0xababab';
            base05 = '0xb9b9b9';
            base06 = '0xe3e3e3';
            base07 = '0xf7f7f7';
            base08 = '0x7c7c7c';
            base09 = '0x999999';
            base0A = '0xa0a0a0';
            base0B = '0x8e8e8e';
            base0C = '0x868686';
            base0D = '0x686868';
            base0E = '0x747474';
            base0F = '0x5e5e5e';
            
            
        case 'greenscreen'
            % Base16 Green Screen
            % Scheme: Chris Kempson (http://chriskempson.com)
            
            base00 = '0x001100';
            base01 = '0x003300';
            base02 = '0x005500';
            base03 = '0x007700';
            base04 = '0x009900';
            base05 = '0x00bb00';
            base06 = '0x00dd00';
            base07 = '0x00ff00';
            base08 = '0x007700';
            base09 = '0x009900';
            base0A = '0x007700';
            base0B = '0x00bb00';
            base0C = '0x005500';
            base0D = '0x009900';
            base0E = '0x00bb00';
            base0F = '0x005500';
            
            
        case 'harmonic16'
            % Base16 harmonic16
            % Scheme: Jannik Siebert (https://github.com/janniks)
            
            base00 = '0x0b1c2c';
            base01 = '0x223b54';
            base02 = '0x405c79';
            base03 = '0x627e99';
            base04 = '0xaabcce';
            base05 = '0xcbd6e2';
            base06 = '0xe5ebf1';
            base07 = '0xf7f9fb';
            base08 = '0xbf8b56';
            base09 = '0xbfbf56';
            base0A = '0x8bbf56';
            base0B = '0x56bf8b';
            base0C = '0x568bbf';
            base0D = '0x8b56bf';
            base0E = '0xbf568b';
            base0F = '0xbf5656';
            
            
        case 'isotope'
            % Base16 Isotope
            % Scheme: Jan T. Sott
            
            base00 = '0x000000';
            base01 = '0x404040';
            base02 = '0x606060';
            base03 = '0x808080';
            base04 = '0xc0c0c0';
            base05 = '0xd0d0d0';
            base06 = '0xe0e0e0';
            base07 = '0xffffff';
            base08 = '0xff0000';
            base09 = '0xff9900';
            base0A = '0xff0099';
            base0B = '0x33ff00';
            base0C = '0x00ffff';
            base0D = '0x0066ff';
            base0E = '0xcc00ff';
            base0F = '0x3300ff';
            
            
        case 'londontube'
            % Base16 London Tube
            % Scheme: Jan T. Sott
            
            base00 = '0x231f20';
            base01 = '0x1c3f95';
            base02 = '0x5a5758';
            base03 = '0x737171';
            base04 = '0x959ca1';
            base05 = '0xd9d8d8';
            base06 = '0xe7e7e8';
            base07 = '0xffffff';
            base08 = '0xee2e24';
            base09 = '0xf386a1';
            base0A = '0xffd204';
            base0B = '0x00853e';
            base0C = '0x85cebc';
            base0D = '0x009ddc';
            base0E = '0x98005d';
            base0F = '0xb06110';
            
            
        case 'marrakesh'
            % Base16 Marrakesh
            % Scheme: Alexandre Gavioli (http://github.com/Alexx2/)
            
            base00 = '0x201602';
            base01 = '0x302e00';
            base02 = '0x5f5b17';
            base03 = '0x6c6823';
            base04 = '0x86813b';
            base05 = '0x948e48';
            base06 = '0xccc37a';
            base07 = '0xfaf0a5';
            base08 = '0xc35359';
            base09 = '0xb36144';
            base0A = '0xa88339';
            base0B = '0x18974e';
            base0C = '0x75a738';
            base0D = '0x477ca1';
            base0E = '0x8868b3';
            base0F = '0xb3588e';
            
            
        case 'mocha'
            % Base16 Mocha
            % Scheme: Chris Kempson (http://chriskempson.com)
            
            base00 = '0x3B3228';
            base01 = '0x534636';
            base02 = '0x645240';
            base03 = '0x7e705a';
            base04 = '0xb8afad';
            base05 = '0xd0c8c6';
            base06 = '0xe9e1dd';
            base07 = '0xf5eeeb';
            base08 = '0xcb6077';
            base09 = '0xd28b71';
            base0A = '0xf4bc87';
            base0B = '0xbeb55b';
            base0C = '0x7bbda4';
            base0D = '0x8ab3b5';
            base0E = '0xa89bb9';
            base0F = '0xbb9584';
            
            
        case 'monokai'
            % Base16 Monokai
            % Scheme: Wimer Hazenberg (http://www.monokai.nl)
            
            base00 = '0x272822';
            base01 = '0x383830';
            base02 = '0x49483e';
            base03 = '0x75715e';
            base04 = '0xa59f85';
            base05 = '0xf8f8f2';
            base06 = '0xf5f4f1';
            base07 = '0xf9f8f5';
            base08 = '0xf92672';
            base09 = '0xfd971f';
            base0A = '0xf4bf75';
            base0B = '0xa6e22e';
            base0C = '0xa1efe4';
            base0D = '0x66d9ef';
            base0E = '0xae81ff';
            base0F = '0xcc6633';
            
            
        case 'ocean'
            % Base16 Ocean
            % Scheme: Chris Kempson (http://chriskempson.com)
            
            base00 = '0x2b303b';
            base01 = '0x343d46';
            base02 = '0x4f5b66';
            base03 = '0x65737e';
            base04 = '0xa7adba';
            base05 = '0xc0c5ce';
            base06 = '0xdfe1e8';
            base07 = '0xeff1f5';
            base08 = '0xbf616a';
            base09 = '0xd08770';
            base0A = '0xebcb8b';
            base0B = '0xa3be8c';
            base0C = '0x96b5b4';
            base0D = '0x8fa1b3';
            base0E = '0xb48ead';
            base0F = '0xab7967';
            
            
        case 'paraiso'
            % Base16 Paraiso
            % Scheme: Jan T. Sott
            
            base00 = '0x2f1e2e';
            base01 = '0x41323f';
            base02 = '0x4f424c';
            base03 = '0x776e71';
            base04 = '0x8d8687';
            base05 = '0xa39e9b';
            base06 = '0xb9b6b0';
            base07 = '0xe7e9db';
            base08 = '0xef6155';
            base09 = '0xf99b15';
            base0A = '0xfec418';
            base0B = '0x48b685';
            base0C = '0x5bc4bf';
            base0D = '0x06b6ef';
            base0E = '0x815ba4';
            base0F = '0xe96ba8';
            
            
        case 'railscasts'
            % Base16 Railscasts
            % Scheme: Ryan Bates (http://railscasts.com)
            
            base00 = '0x2b2b2b';
            base01 = '0x272935';
            base02 = '0x3a4055';
            base03 = '0x5a647e';
            base04 = '0xd4cfc9';
            base05 = '0xe6e1dc';
            base06 = '0xf4f1ed';
            base07 = '0xf9f7f3';
            base08 = '0xda4939';
            base09 = '0xcc7833';
            base0A = '0xffc66d';
            base0B = '0xa5c261';
            base0C = '0x519f50';
            base0D = '0x6d9cbe';
            base0E = '0xb6b3eb';
            base0F = '0xbc9458';
            
            
        case 'shapeshifter'
            % Base16 shapeshifter
            % Scheme: Tyler Benziger (http://tybenz.com)
            
            base00 = '0x000000';
            base01 = '0x040404';
            base02 = '0x102015';
            base03 = '0x343434';
            base04 = '0x555555';
            base05 = '0xababab';
            base06 = '0xe0e0e0';
            base07 = '0xf9f9f9';
            base08 = '0xe92f2f';
            base09 = '0xe09448';
            base0A = '0xdddd13';
            base0B = '0x0ed839';
            base0C = '0x23edda';
            base0D = '0x3b48e3';
            base0E = '0xf996e2';
            base0F = '0x69542d';
            
            
        case 'solarized'
            % Base16 Solarized
            % Scheme: Ethan Schoonover (http://ethanschoonover.com/solarized)
            
            base00 = '0x002b36';
            base01 = '0x073642';
            base02 = '0x586e75';
            base03 = '0x657b83';
            base04 = '0x839496';
            base05 = '0x93a1a1';
            base06 = '0xeee8d5';
            base07 = '0xfdf6e3';
            base08 = '0xdc322f';
            base09 = '0xcb4b16';
            base0A = '0xb58900';
            base0B = '0x859900';
            base0C = '0x2aa198';
            base0D = '0x268bd2';
            base0E = '0x6c71c4';
            base0F = '0xd33682';
            
            
        case 'summerfruit'
            % Base16 Summerfruit
            % Scheme: Christopher Corley (http://cscorley.github.io/)
            
            base00 = '0x151515';
            base01 = '0x202020';
            base02 = '0x303030';
            base03 = '0x505050';
            base04 = '0xB0B0B0';
            base05 = '0xD0D0D0';
            base06 = '0xE0E0E0';
            base07 = '0xFFFFFF';
            base08 = '0xFF0086';
            base09 = '0xFD8900';
            base0A = '0xABA800';
            base0B = '0x00C918';
            base0C = '0x1faaaa';
            base0D = '0x3777E6';
            base0E = '0xAD00A1';
            base0F = '0xcc6633';
            
            
        case 'tomorrow'
            % Base16 Tomorrow
            % Scheme: Chris Kempson (http://chriskempson.com)
            
            base00 = '0x1d1f21';
            base01 = '0x282a2e';
            base02 = '0x373b41';
            base03 = '0x969896';
            base04 = '0xb4b7b4';
            base05 = '0xc5c8c6';
            base06 = '0xe0e0e0';
            base07 = '0xffffff';
            base08 = '0xcc6666';
            base09 = '0xde935f';
            base0A = '0xf0c674';
            base0B = '0xb5bd68';
            base0C = '0x8abeb7';
            base0D = '0x81a2be';
            base0E = '0xb294bb';
            base0F = '0xa3685a';
            
            
        otherwise
            str = [];
            str = [str, sprintf('Color scheme not found: %s.', name)];
            str = [str, sprintf('Using default scheme ''solarized-dark''')];
            warning(str);
            
            b    = get_base16_('solarized');
            flag = 1;
            return
            
    end
    
    b = struct(...
        'x00', base00, ...
        'x01', base01, ...
        'x02', base02, ...
        'x03', base03, ...
        'x04', base04, ...
        'x05', base05, ...
        'x06', base06, ...
        'x07', base07, ...
        'x08', base08, ...
        'x09', base09, ...
        'x0A', base0A, ...
        'x0B', base0B, ...
        'x0C', base0C, ...
        'x0D', base0D, ...
        'x0E', base0E, ...
        'x0F', base0F);
    
end
