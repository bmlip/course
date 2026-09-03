%!PS-Adobe-3.0 EPSF-3.0
%%Creator: Mayura Draw, Version 4.3
%%Title: fig-classification-generative-4.md
%%CreationDate: Fri Dec 05 13:48:09 2008
%%BoundingBox: 173 361 475 604
%%DocumentFonts: ArialMT
%%+ SymbolMT
%%+ Arial-ItalicMT
%%Orientation: Portrait
%%EndComments
%%BeginProlog
%%BeginResource: procset MayuraDraw_ops
%%Version: 4.3
%%Copyright: (c) 1993-2003 Mayura Software
/PDXDict 100 dict def
PDXDict begin
% width height matrix proc key cache
% definepattern -\> font
/definepattern { %def
  7 dict begin
    /FontDict 9 dict def
    FontDict begin
      /cache exch def
      /key exch def
      /proc exch cvx def
      /mtx exch matrix invertmatrix def
      /height exch def
      /width exch def
      /ctm matrix currentmatrix def
      /ptm matrix identmatrix def
      /str
      (xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx)
      def
    end
    /FontBBox [ %def
      0 0 FontDict /width get
      FontDict /height get
    ] def
    /FontMatrix FontDict /mtx get def
    /Encoding StandardEncoding def
    /FontType 3 def
    /BuildChar { %def
      pop begin
      FontDict begin
        width 0 cache { %ifelse
          0 0 width height setcachedevice
        }{ %else
          setcharwidth
        } ifelse
        0 0 moveto width 0 lineto
        width height lineto 0 height lineto
        closepath clip newpath
        gsave proc grestore
      end end
    } def
    FontDict /key get currentdict definefont
  end
} bind def

% dict patternpath -
% dict matrix patternpath -
/patternpath { %def
  dup type /dicttype eq { %ifelse
    begin FontDict /ctm get setmatrix
  }{ %else
    exch begin FontDict /ctm get setmatrix
    concat
  } ifelse
  currentdict setfont
  FontDict begin
    FontMatrix concat
    width 0 dtransform
    round width div exch round width div exch
    0 height dtransform
    round height div exch
    round height div exch
    0 0 transform round exch round exch
    ptm astore setmatrix

    pathbbox
    height div ceiling height mul 4 1 roll
    width div ceiling width mul 4 1 roll
    height div floor height mul 4 1 roll
    width div floor width mul 4 1 roll

    2 index sub height div ceiling cvi exch
    3 index sub width div ceiling cvi exch
    4 2 roll moveto

    FontMatrix ptm invertmatrix pop
    { %repeat
      gsave
        ptm concat
        dup str length idiv { %repeat
          str show
        } repeat
        dup str length mod str exch
        0 exch getinterval show
      grestore
      0 height rmoveto
    } repeat
    pop
  end end
} bind def

% dict patternfill -
% dict matrix patternfill -
/patternfill { %def
  gsave
    eoclip patternpath
  grestore
  newpath
} bind def

/img { %def
  gsave
  /imgh exch def
  /imgw exch def
  concat
  imgw imgh 8
  [imgw 0 0 imgh neg 0 imgh]
  /colorstr 768 string def
  /colorimage where {
    pop
    { currentfile colorstr readhexstring pop }
    false 3 colorimage
  }{
    /graystr 256 string def
    {
      currentfile colorstr readhexstring pop
      length 3 idiv
      dup 1 sub 0 1 3 -1 roll
      {
        graystr exch
        colorstr 1 index 3 mul get 30 mul
        colorstr 2 index 3 mul 1 add get 59 mul
        colorstr 3 index 3 mul 2 add get 11 mul
        add add 100 idiv
        put
      } for
      graystr 0 3 -1 roll getinterval
    } image
  } ifelse
  grestore
} bind def

/arrowhead {
  gsave
    [] 0 setdash
    strokeC strokeM strokeY strokeK setcmykcolor
    2 copy moveto
    4 2 roll exch 4 -1 roll exch
    sub 3 1 roll sub
    exch atan rotate dup scale
    arrowtype
    dup 0 eq {
      -1 2 rlineto 7 -2 rlineto -7 -2 rlineto
      closepath fill
    } if
    dup 1 eq {
      0 3 rlineto 9 -3 rlineto -9 -3 rlineto
      closepath fill
    } if
    dup 2 eq {
      -6 -6 rmoveto 6 6 rlineto -6 6 rlineto
      -1.4142 -1.4142 rlineto 4.5858 -4.5858 rlineto
      -4.5858 -4.5858 rlineto closepath fill
    } if
    dup 3 eq {
      -6 0 rmoveto -1 2 rlineto 7 -2 rlineto -7 -2 rlineto
      closepath fill
    } if
    dup 4 eq {
      -9 0 rmoveto 0 3 rlineto 9 -3 rlineto -9 -3 rlineto
      closepath fill
    } if
    dup 5 eq {
      currentpoint newpath 3 0 360 arc
      closepath fill
    } if
    dup 6 eq {
      2.5 2.5 rmoveto 0 -5 rlineto -5 0 rlineto 0 5 rlineto
      closepath fill
    } if
    pop
  grestore
} bind def

/setcmykcolor where { %ifelse
  pop
}{ %else
  /setcmykcolor {
     /black exch def /yellow exch def
     /magenta exch def /cyan exch def
     cyan black add dup 1 gt { pop 1 } if 1 exch sub
     magenta black add dup 1 gt { pop 1 } if 1 exch sub
     yellow black add dup 1 gt { pop 1 } if 1 exch sub
     setrgbcolor
  } bind def
} ifelse

/RE { %def
  findfont begin
  currentdict dup length dict begin
    { %forall
      1 index /FID ne { def } { pop pop } ifelse
    } forall
    /FontName exch def dup length 0 ne { %if
      /Encoding Encoding 256 array copy def
      0 exch { %forall
        dup type /nametype eq { %ifelse
          Encoding 2 index 2 index put
          pop 1 add
        }{ %else
          exch pop
        } ifelse
      } forall
    } if pop
  currentdict dup end end
  /FontName get exch definefont pop
} bind def

/spacecount { %def
  0 exch
  ( ) { %loop
    search { %ifelse
      pop 3 -1 roll 1 add 3 1 roll
    }{ pop exit } ifelse
  } loop
} bind def

/WinAnsiEncoding [
  39/quotesingle 96/grave 130/quotesinglbase/florin/quotedblbase
  /ellipsis/dagger/daggerdbl/circumflex/perthousand
  /Scaron/guilsinglleft/OE 145/quoteleft/quoteright
  /quotedblleft/quotedblright/bullet/endash/emdash
  /tilde/trademark/scaron/guilsinglright/oe/dotlessi
  159/Ydieresis 164/currency 166/brokenbar 168/dieresis/copyright
  /ordfeminine 172/logicalnot 174/registered/macron/ring
  177/plusminus/twosuperior/threesuperior/acute/mu
  183/periodcentered/cedilla/onesuperior/ordmasculine
  188/onequarter/onehalf/threequarters 192/Agrave/Aacute
  /Acircumflex/Atilde/Adieresis/Aring/AE/Ccedilla
  /Egrave/Eacute/Ecircumflex/Edieresis/Igrave/Iacute
  /Icircumflex/Idieresis/Eth/Ntilde/Ograve/Oacute
  /Ocircumflex/Otilde/Odieresis/multiply/Oslash
  /Ugrave/Uacute/Ucircumflex/Udieresis/Yacute/Thorn
  /germandbls/agrave/aacute/acircumflex/atilde/adieresis
  /aring/ae/ccedilla/egrave/eacute/ecircumflex
  /edieresis/igrave/iacute/icircumflex/idieresis
  /eth/ntilde/ograve/oacute/ocircumflex/otilde
  /odieresis/divide/oslash/ugrave/uacute/ucircumflex
  /udieresis/yacute/thorn/ydieresis
] def

/SymbolEncoding [
  32/space/exclam/universal/numbersign/existential/percent
  /ampersand/suchthat/parenleft/parenright/asteriskmath/plus
  /comma/minus/period/slash/zero/one/two/three/four/five/six
  /seven/eight/nine/colon/semicolon/less/equal/greater/question
  /congruent/Alpha/Beta/Chi/Delta/Epsilon/Phi/Gamma/Eta/Iota
  /theta1/Kappa/Lambda/Mu/Nu/Omicron/Pi/Theta/Rho/Sigma/Tau
  /Upsilon/sigma1/Omega/Xi/Psi/Zeta/bracketleft/therefore
  /bracketright/perpendicular/underscore/radicalex/alpha
  /beta/chi/delta/epsilon/phi/gamma/eta/iota/phi1/kappa/lambda
  /mu/nu/omicron/pi/theta/rho/sigma/tau/upsilon/omega1/omega
  /xi/psi/zeta/braceleft/bar/braceright/similar
  161/Upsilon1/minute/lessequal/fraction/infinity/florin/club
  /diamond/heart/spade/arrowboth/arrowleft/arrowup/arrowright
  /arrowdown/degree/plusminus/second/greaterequal/multiply
  /proportional/partialdiff/bullet/divide/notequal/equivalence
  /approxequal/ellipsis/arrowvertex/arrowhorizex/carriagereturn
  /aleph/Ifraktur/Rfraktur/weierstrass/circlemultiply
  /circleplus/emptyset/intersection/union/propersuperset
  /reflexsuperset/notsubset/propersubset/reflexsubset/element
  /notelement/angle/gradient/registerserif/copyrightserif
  /trademarkserif/product/radical/dotmath/logicalnot/logicaland
  /logicalor/arrowdblboth/arrowdblleft/arrowdblup/arrowdblright
  /arrowdbldown/lozenge/angleleft/registersans/copyrightsans
  /trademarksans/summation/parenlefttp/parenleftex/parenleftbt
  /bracketlefttp/bracketleftex/bracketleftbt/bracelefttp
  /braceleftmid/braceleftbt/braceex
  241/angleright/integral/integraltp/integralex/integralbt
  /parenrighttp/parenrightex/parenrightbt/bracketrighttp
  /bracketrightex/bracketrightbt/bracerighttp/bracerightmid
  /bracerightbt
] def

/patarray [
/leftdiagonal /rightdiagonal /crossdiagonal /horizontal
/vertical /crosshatch /fishscale /wave /brick
] def
/arrowtype 0 def
/fillC 0 def /fillM 0 def /fillY 0 def /fillK 0 def
/strokeC 0 def /strokeM 0 def /strokeY 0 def /strokeK 1 def
/pattern -1 def
/mat matrix def
/mat2 matrix def
/nesting 0 def
/deferred /N def
/c /curveto load def
/c2 { pop pop c } bind def
/C /curveto load def
/C2 { pop pop C } bind def
/e { gsave concat 0 0 moveto } bind def
/F {
  nesting 0 eq { %ifelse
    pattern -1 eq { %ifelse
      fillC fillM fillY fillK setcmykcolor eofill
    }{ %else
      gsave fillC fillM fillY fillK setcmykcolor eofill grestore
      0 0 0 1 setcmykcolor
      patarray pattern get findfont patternfill
    } ifelse
  }{ %else
    /deferred /F def
  } ifelse
} bind def
/f { closepath F } bind def
/K { /strokeK exch def /strokeY exch def
     /strokeM exch def /strokeC exch def } bind def
/k { /fillK exch def /fillY exch def
     /fillM exch def /fillC exch def } bind def
/opc { pop } bind def
/Opc { pop } bind def
/L /lineto load def
/L2 { pop pop L } bind def
/m /moveto load def
/m2 { pop pop m } bind def
/n /newpath load def
/N {
  nesting 0 eq { %ifelse
    newpath
  }{ %else
    /deferred /N def
  } ifelse
} def
/S {
  nesting 0 eq { %ifelse
    strokeC strokeM strokeY strokeK setcmykcolor stroke
  }{ %else
    /deferred /S def
  } ifelse
} bind def
/s { closepath S } bind def
/Tx { fillC fillM fillY fillK setcmykcolor show
      0 leading neg translate 0 0 moveto } bind def
/T { grestore } bind def
/TX { pop } bind def
/Ts { pop } bind def
/tal { pop } bind def
/tld { pop } bind def
/tbx { pop exch pop sub /jwidth exch def } def
/tpt { %def
  fillC fillM fillY fillK setcmykcolor
  moveto show
} bind def
/tpj { %def
  fillC fillM fillY fillK setcmykcolor
  moveto
  dup stringwidth pop
  3 -1 roll
  exch sub
  1 index spacecount
  dup 0 eq { %ifelse
    pop pop show
  }{ %else
    div 0 8#040 4 -1 roll widthshow
  } ifelse
} bind def
/u {} def
/U {} def
/*u { /nesting nesting 1 add def } def
/*U {
  /nesting nesting 1 sub def
  nesting 0 eq {
    deferred cvx exec
  } if
} def
/w /setlinewidth load def
/d /setdash load def
/B {
  nesting 0 eq { %ifelse
    gsave F grestore S
  }{ %else
    /deferred /B def
  } ifelse
} bind def
/b { closepath B } bind def
/z { /align exch def pop /leading exch def exch findfont
     exch scalefont setfont } bind def
/tfn { exch findfont
     exch scalefont setfont } bind def
/Pat { /pattern exch def } bind def
/cm { 6 array astore concat } bind def
/q { mat2 currentmatrix pop } bind def
/Q { mat2 setmatrix } bind def
/Ah {
  pop /arrowtype exch def
  currentlinewidth 5 1 roll arrowhead
} bind def
/Arc {
  mat currentmatrix pop
    translate scale 0 0 1 5 -2 roll arc
  mat setmatrix
} bind def
/Arc2 { pop pop Arc } bind def
/Bx {
  mat currentmatrix pop
    concat /y1 exch def /x1 exch def /y2 exch def /x2 exch def
    x1 y1 moveto x1 y2 lineto x2 y2 lineto x2 y1 lineto
  mat setmatrix
} bind def
/Rr {
  mat currentmatrix pop
    concat /yrad exch def /xrad exch def
    2 copy gt { exch } if /x2 exch def /x1 exch def
    2 copy gt { exch } if /y2 exch def /y1 exch def
    x1 xrad add y2 moveto
    matrix currentmatrix x1 xrad add y2 yrad sub translate xrad yrad scale
    0 0 1 90 -180 arc setmatrix
    matrix currentmatrix x1 xrad add y1 yrad add translate xrad yrad scale
    0 0 1 180 270 arc setmatrix
    matrix currentmatrix x2 xrad sub y1 yrad add translate xrad yrad scale
    0 0 1 270 0 arc setmatrix
    matrix currentmatrix x2 xrad sub y2 yrad sub translate xrad yrad scale
    0 0 1 0 90 arc setmatrix
    closepath
  mat setmatrix
} bind def
/Ov {
  mat currentmatrix pop
    concat translate scale 1 0 moveto 0 0 1 0 360 arc closepath
  mat setmatrix
} bind def
end
%%EndResource
%%EndProlog
%%BeginSetup
%PDX g 3 3 1 1
%%IncludeFont: ArialMT
%%IncludeFont: SymbolMT
%%IncludeFont: Arial-ItalicMT
PDXDict begin
%%EndSetup
%%Page: 1 1
%%BeginPageSetup
/_PDX_savepage save def

15 15 [300 72 div 0 0 300 72 div 0 0]
{ %definepattern
  2 setlinecap
  7.5 0 moveto 15 7.5 lineto
  0 7.5 moveto 7.5 15 lineto
  2 setlinewidth stroke
} bind
/rightdiagonal true definepattern pop

15 15 [300 72 div 0 0 300 72 div 0 0]
{ %definepattern
  2 setlinecap
  7.5 0 moveto 0 7.5 lineto
  15 7.5 moveto 7.5 15 lineto
  2 setlinewidth stroke
} bind
/leftdiagonal true definepattern pop

15 15 [300 72 div 0 0 300 72 div 0 0]
{ %definepattern
  2 setlinecap
  0 7.5 moveto 15 7.5 lineto
  2 setlinewidth stroke
} bind
/horizontal true definepattern pop

15 15 [300 72 div 0 0 300 72 div 0 0]
{ %definepattern
  2 setlinecap
  7.5 0 moveto 7.5 15 lineto
  2 setlinewidth stroke
} bind
/vertical true definepattern pop

15 15 [300 72 div 0 0 300 72 div 0 0]
{ %definepattern
  2 setlinecap
  0 7.5 moveto 15 7.5 lineto
  7.5 0 moveto 7.5 15 lineto
  2 setlinewidth stroke
} bind
/crosshatch true definepattern pop

30 30 [300 72 div 0 0 300 72 div 0 0]
{ %definepattern
  2 setlinecap
  0 7.5 moveto 30 7.5 lineto
  0 22.5 moveto 30 22.5 lineto
  7.5 0 moveto 7.5 7.5 lineto
  7.5 22.5 moveto 7.5 30 lineto
  22.5 7.5 moveto 22.5 22.5 lineto
  1 setlinewidth stroke
} bind
/brick true definepattern pop

30 30 [300 72 div 0 0 300 72 div 0 0]
{ %definepattern
  2 2 scale
  2 setlinecap
  7.5 0 moveto 15 7.5 lineto
  0 7.5 moveto 7.5 15 lineto
  7.5 0 moveto 0 7.5 lineto
  15 7.5 moveto 7.5 15 lineto
  0.5 setlinewidth stroke
} bind
/crossdiagonal true definepattern pop

30 30 [300 72 div 0 0 300 72 div 0 0]
{ %definepattern
  2 2 scale
  1 setlinecap
  0 7.5 moveto 0 15 7.5 270 360 arc
  7.5 15 moveto 15 15 7.5 180 270 arc
  0 7.5 moveto 7.5 7.5 7.5 180 360 arc
  0.5 setlinewidth stroke
} bind
/fishscale true definepattern pop

30 30 [300 72 div 0 0 300 72 div 0 0]
{ %definepattern
  1 setlinecap 0.5 setlinewidth
  7.5 0 10.6 135 45 arcn
  22.5 15 10.6 225 315 arc
  stroke
  7.5 15 10.6 135 45 arcn
  22.5 30 10.6 225 315 arc
  stroke
} bind
/wave true definepattern pop

WinAnsiEncoding /_ArialMT /ArialMT RE
SymbolEncoding /_SymbolMT /SymbolMT RE
WinAnsiEncoding /_Arial-ItalicMT /Arial-ItalicMT RE

newpath 2 setlinecap 0 setlinejoin 2 setmiterlimit
[] 0 setdash
173 361 moveto 173 604 lineto 475 604 lineto 475 361 lineto closepath clip
newpath
%%EndPageSetup
0.5 w
q
1 0 0 1 24 9 cm
165 572 165 573 m2
165 369 L
Q
S
q
1 0 0 1 24 9 cm
165 369 165 573 2 1 Ah
Q
q
1 0 0 1 3 42 cm
180 345 m
440 345 441 345 L2
Q
S
q
1 0 0 1 3 42 cm
180 345 441 345 2 2 Ah
Q
[1 0 0 1 0 75] e
420 330 420 330 tbx
0 tal
15 tld
1 1 1 0 k
/_ArialMT 14 tfn
(x) 420 317.33 tpt
-3 Ts
/_ArialMT 8 tfn
(1) 427 314.33 tpt
T
[1 0 0 1 -228 240] e
420 330 420 330 tbx
0 tal
15 tld
0 Ts
/_ArialMT 14 tfn
(x) 420 317.33 tpt
-3 Ts
/_ArialMT 8 tfn
(2) 427 314.33 tpt
T
u
u
-1.42109e-016 -1.42109e-016 -1.42109e-016 0 k
-1.42109e-016 1 1 0 K
q
1 0 0 1 12 -51 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 15 -51 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 15 -51 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 36 -66 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 39 -66 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 39 -66 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 15 -30 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 18 -30 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 18 -30 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 39 -39 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 42 -39 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 42 -39 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 72 -63 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 75 -63 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 75 -63 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 57 -54 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 60 -54 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 60 -54 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 93 -18 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 96 -18 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 96 -18 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 69 -6 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 72 -6 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 72 -6 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 120 -39 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 123 -39 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 123 -39 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 57 -93 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 60 -93 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 60 -93 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 42 -18 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 45 -18 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 45 -18 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 87 -39 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 90 -39 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 90 -39 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 75 -24 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 78 -24 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 78 -24 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 21 -57 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 24 -57 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 24 -57 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 45 -72 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 48 -72 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 48 -72 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 33 -27 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 36 -27 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 36 -27 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 48 -45 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 51 -45 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 51 -45 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 120 -66 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 123 -66 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 123 -66 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 66 -42 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 69 -42 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 69 -42 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 102 -24 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 105 -24 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 105 -24 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 129 -45 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 132 -45 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 132 -45 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 51 -24 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 54 -24 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 54 -24 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 96 -45 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 99 -45 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 99 -45 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 78 -36 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 81 -36 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 81 -36 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 18 -39 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 21 -39 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 21 -39 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 42 -54 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 45 -54 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 45 -54 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 21 -18 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 24 -18 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 24 -18 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 45 -27 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 48 -27 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 48 -27 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 78 -51 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 81 -51 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 81 -51 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 63 -21 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 66 -21 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 66 -21 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 99 -6 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 102 -6 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 102 -6 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 144 -39 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 147 -39 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 147 -39 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 48 -6 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 51 -6 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 51 -6 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 120 0 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 123 0 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 123 0 cm
198 501 m
201 501 L
Q
S
U
u
q
1 0 0 1 84 0 cm
198 507 m
198 495 L
Q
S
q
1 0 0 1 87 0 cm
189 501 m
198 501 L
Q
S
q
1 0 0 1 87 0 cm
198 501 m
201 501 L
Q
S
U
U
1 1 -1.42109e-016 0 K
4.24264 4.24264 393 546 [1 0 0 1 -87 27] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -99 -6] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -111 -24] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -120 15] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -117 -3] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -90 -30] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -108 -12] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -117 -27] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -123 -15] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -96 -12] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -93 -30] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -126 3] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -129 -27] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -132 15] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -108 -45] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -141 0] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -147 12] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -147 -30] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -132 -45] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -96 -60] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -135 -27] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -123 -57] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -111 -54] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -138 3] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -147 -12] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -153 0] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -153 -42] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -141 -39] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -123 -63] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -126 -26.58] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -138 -44.58] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -147 -5.577] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -144 -23.58] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -117 -50.58] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -135 -32.58] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -144 -47.58] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -150 -35.58] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -123 -32.58] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -120 -50.58] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -153 -17.58] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -156 -47.58] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -101.4 -22.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -113.4 -40.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -122.4 -1.761] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -119.4 -19.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -92.39 -46.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -110.4 -28.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -119.4 -43.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -125.4 -31.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -98.39 -28.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -95.39 -46.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -128.4 -13.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -131.4 -43.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -113.4 -4.761] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -125.4 -22.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -134.4 16.24] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -131.4 -1.761] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -104.4 -28.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -122.4 -10.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -131.4 -25.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -137.4 -13.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -110.4 -10.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -107.4 -28.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -140.4 4.239] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -143.4 -25.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -77.39 -10.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -89.39 -28.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -98.39 10.24] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -95.39 -7.761] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -68.39 -34.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -86.39 -16.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -95.39 -31.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -101.4 -19.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -74.39 -16.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -71.39 -34.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -104.4 -1.761] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -107.4 -31.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -86.39 -22.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -98.39 -40.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -107.4 -1.761] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -104.4 -19.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -77.39 -46.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -95.39 -28.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -104.4 -43.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -110.4 -31.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -83.39 -28.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -80.39 -46.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -113.4 -13.76] Ov
s
4.24264 4.24264 393 546 [1 0 0 1 -116.4 -43.76] Ov
s
[1 0 0 1 -57 -180] e
309 567 309 567 tbx
0 tal
15 tld
0 Ts
1 1 1 0 k
/_ArialMT 14 tfn
() 309 554.33 tpt
T
[1 0 0 1 -71.29 68.27] e
411 444 411 444 tbx
0 tal
15 tld
/_ArialMT 12 tfn
(p\() 411 433.14 tpt
1 1 -1.42109e-016 0 k
(y=b) 421.668 433.14 tpt
1 1 1 0 k
(|x) 441.348 433.14 tpt
-5 Ts
/_ArialMT 8 tfn
(new) 450.468 428.14 tpt
-2 Ts
/_ArialMT 14 tfn
(\) = ) 465.14 431.14 tpt
-1.42109e-016 1 1 0 k
() 485.762 431.14 tpt
(\r) TX
(                ) 411 416.14 tpt
T
1 1 1 0 k
1 1 1 0 K
4.24264 4.24264 393 546 [1 0 0 1 -69.39 -54.51] Ov
b
[1 0 0 1 36 159] e
282 402 282 402 tbx
0 tal
15 tld
0 Ts
1 1 -1.42109e-016 0 k
/_ArialMT 14 tfn
(p\(y=`b'\) = ) 282 389.33 tpt
/_SymbolMT 14 tfn
(p) 345.364 389.33 tpt
-3 Ts
/_ArialMT 9 tfn
(b) 353.05 386.33 tpt
T
[1 0 0 1 0 27] e
282 402 282 402 tbx
0 tal
15 tld
0 Ts
-1.42109e-016 1 1 0 k
/_ArialMT 14 tfn
(p\(y=`r'\) = ) 282 389.33 tpt
/_SymbolMT 14 tfn
(p) 342.242 389.33 tpt
-3 Ts
/_ArialMT 9 tfn
(r) 349.928 386.33 tpt
T
[1 0 0 1 -3 219] e
222 375 222 375 tbx
0 tal
12 tld
0 Ts
1 1 1 0 k
/_ArialMT 11 tfn
(3. Use Bayes to calc. posterior class prob.) 222 365.045 tpt
T
[1 0 0 1 45 138] e
282 402 282 402 tbx
0 tal
15 tld
1 1 -1.42109e-016 0 k
/_ArialMT 14 tfn
(p\(x|y=`b'\) = ) 282 389.33 tpt
/_Arial-ItalicMT 14 tfn
(N) 356.004 389.33 tpt
/_ArialMT 14 tfn
(\(x|) 366.112 389.33 tpt
/_SymbolMT 14 tfn
(m) 381.414 389.33 tpt
-3 Ts
/_ArialMT 9 tfn
(b) 389.478 386.33 tpt
0 Ts
/_ArialMT 14 tfn
(,) 394.482 389.33 tpt
/_SymbolMT 14 tfn
(S) 398.374 389.33 tpt
/_ArialMT 14 tfn
(\)) 406.662 389.33 tpt
T
-1.42109e-016 -1.42109e-016 -1.42109e-016 0 k
-1.42109e-016 1 1 0 K
2 w
50.9117 38.1838 225 489 [1 0 0 1 33 -30] Ov
s
1 1 -1.42109e-016 0 K
50.9117 38.1838 225 489 [1 0 0 1 42 36] Ov
s
[1 0 0 1 39 48] e
282 402 282 402 tbx
0 tal
15 tld
-1.42109e-016 1 1 0 k
/_ArialMT 14 tfn
(p\(x|y=`r'\) = ) 282 389.33 tpt
/_Arial-ItalicMT 14 tfn
(N) 352.882 389.33 tpt
/_ArialMT 14 tfn
(\(x|) 362.99 389.33 tpt
/_SymbolMT 14 tfn
(m) 378.292 389.33 tpt
-3 Ts
/_ArialMT 9 tfn
(r) 386.356 386.33 tpt
0 Ts
/_ArialMT 14 tfn
(,) 389.353 389.33 tpt
/_SymbolMT 14 tfn
(S) 393.245 389.33 tpt
/_ArialMT 14 tfn
(\)) 401.533 389.33 tpt
T
u
[0.72 0 0 0.72 103.9 44.06] e
360 627 360 627 tbx
0 tal
12 tld
1 1 -1.42109e-016 0 k
/_ArialMT 11 tfn
(p\(x) 360 617.045 tpt
-3 Ts
/_ArialMT 8 tfn
(new) 375.279 614.045 tpt
0 Ts
/_ArialMT 11 tfn
(|y=b\)p\(y=b\)) 389.951 617.045 tpt
-1.42109e-016 1 1 0 k
() 445.996 617.045 tpt
(\r) TX
() 360 605.045 tpt
T
[0.72 0 0 0.72 73.68 33.26] e
360 627 360 627 tbx
0 tal
12 tld
1 1 -1.42109e-016 0 k
/_ArialMT 11 tfn
(p\(x) 360 617.045 tpt
-3 Ts
/_ArialMT 8 tfn
(new) 375.279 614.045 tpt
0 Ts
/_ArialMT 11 tfn
(|y=b\)p\(y=b\) +) 389.951 617.045 tpt
-1.42109e-016 1 1 0 k
( p\(x) 455.478 617.045 tpt
-3 Ts
/_ArialMT 8 tfn
(new) 473.815 614.045 tpt
0 Ts
/_ArialMT 11 tfn
(|y=r\)p\(y=r\)) 488.487 617.045 tpt
(\r) TX
() 360 605.045 tpt
T
1 1 1 0 k
0.937255 0.937255 0.937255 0 K
0.5 w
q
0.72 0 0 0.72 205.4 11.95 cm
180 657 m
354 657 L
Q
B
U
-1.42109e-016 -1.42109e-016 -1.42109e-016 0 k
465 471 330 513 [1 0 0 1 0 0] Bx
s
%%PageTrailer
_PDX_savepage restore
%%Trailer
end
showpage
%%EOF
