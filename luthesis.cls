\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{luthesis}[2020/07/01 Class for thesis preparation]
\LoadClass[10pt,openany]{book}

% ------------------------------------------------------

\RequirePackage{xcolor}
\definecolor{LUgolden}{RGB}{156,97,20}
\definecolor{LUbla}{RGB}{17,55,125}
%\definecolor{LUgray}{RGB}{13, 2, 79}
\definecolor{LUgray}{RGB}{0, 0, 0}

\RequirePackage{ifthen}

% ------------------------------------------------------

\newboolean{THESIS}\setboolean{THESIS}{false}
\newboolean{Afour}\setboolean{Afour}{false}

\newboolean{CROP}\setboolean{CROP}{false}

\newboolean{DistancesEquations}\setboolean{DistancesEquations}{true}

\newboolean{RM}\setboolean{RM}{false}
\newboolean{SF}\setboolean{SF}{false}

\newboolean{swedish}\setboolean{swedish}{false}
\newboolean{english}\setboolean{english}{false}


% ------------------------------------------------------

\DeclareOption{thesis}{\setboolean{THESIS}{true}}
\DeclareOption{a4}{\setboolean{Afour}{true}}

\DeclareOption{nocrop}{\setboolean{CROP}{false}}
\DeclareOption{crop}{\setboolean{CROP}{true}}

\DeclareOption{rm}{\setboolean{RM}{true}}
\DeclareOption{sf}{\setboolean{SF}{true}}

\DeclareOption{nomathskip}{\setboolean{DistancesEquations}{false}}

\DeclareOption{swedish}{
	\setboolean{english}{false}
	\setboolean{swedish}{true}
}

\DeclareOption{english}{
	\setboolean{swedish}{false}
	\setboolean{english}{true}
	}

% ------------------------------------------------------

\DeclareOption*{\ClassWarning{luthesis}{Cannot process the option: ?\CurrentOption?}}

% ------------------------------------------------------

\ProcessOptions\relax

% ------------------------------------------------------
% ------------------------------------------------------

\RequirePackage{graphicx}
\RequirePackage{calc}
%\RequirePackage{hyperref}

% ------------------------------------------------------

\ifthenelse{\boolean{THESIS}}{%
  \renewcommand\maketitle{
  \begin{titlepage}%
  \let\footnotesize\small
  \let\footnoterule\relax
  \let \footnote \thanks
  \null\vfil
  \begin{center}%
  \vskip -100\p@
  {\includegraphics[width=2.5cm]{./logos/LundUniversity_C2line_BLACK}}
  \vskip 100\p@
	\rule{\textwidth}{1pt}\par
    \vskip 1.5em%
    {\Huge\bfseries \@title \par}%
    \vskip 1.5em%
	\rule{\textwidth}{1pt}\par    
    {\large
     \lineskip .75em%
	  \vskip 60\p@     
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
	  \vskip 60\p@     
%    {\large \@date \par}%       % Set date in \large size.
%  \@thanks
  \vfil\null
  \end{center}\par
  \end{titlepage}%
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}}{}


\ifthenelse{\boolean{Afour}}{%
  \renewcommand\maketitle{
  \begin{titlepage}%
  \let\footnotesize\small
  \let\footnoterule\relax
  \let \footnote \thanks
  \null\vfil
  \begin{center}%
	\vskip 0\p@
	\includegraphics[width=8.0cm]{./logos/Lund_university_L_BLACK}\\[-.25cm]
	\vskip 200\p@
    {\Huge\bfseries \@title \par}%
    {\large
     \lineskip .75em%
	  \vskip 150\p@     
      \begin{tabular}[t]{c}%
        \@author\par
      \end{tabular}\par}%
%    {\large \@date \par}% 
  \vfil\null
  \end{center}\par
  \end{titlepage}%
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}}{}


% ---------------------------------------------------------------------
% TYPEFACES

\ifTHESIS
	\renewcommand{\normalsize}{\fontsize{10.5pt}{12.5pt}\selectfont}
\fi

\ifAfour
	\renewcommand{\normalsize}{\fontsize{12pt}{14pt}\selectfont}
\fi

% ---------------------------------------------------------------------

\ifSF
	\renewcommand{\familydefault}{\sfdefault}
\else
	\renewcommand{\familydefault}{\rmdefault}
\fi
	

% ---------------------------------------------------------------------
% PAGE FORMAT

\RequirePackage{geometry}

\ifAfour
	
	%\geometry{ 
	%	a4paper, twoside = false,         	 	
	%	hmargin = {2.25cm, 2.25cm}, 	
	%	vmargin = {1.25cm, 1.25cm}, 
	%	includehead, includefoot, 	
	%	headsep = 1.0cm,          	
	%	footskip = 1.5cm,        	
	%	headheight = 15pt
	%	}

    \geometry{ 
	   a4paper, twoside = true,         	 	
	   hmargin = {3.0cm, 2.5cm}, 	
	   vmargin = {2.0cm, 2.0cm}, 
	   includehead, includefoot, 	
	   headsep = 1.0cm,          	
	   footskip = 2.0cm,        	
	   headheight = 15pt
	}

\else % thesis

	\newlength{\marksIndent}

	\ifCROP
 		\setlength{\marksIndent}{3mm}
	\else
 		\setlength{\marksIndent}{0mm}
	\fi	
		
	\newlength{\widthPublicatio}
	\setlength{\widthPublicatio}{17cm + 2\marksIndent}
	
	\newlength{\heightPublicatio}
	\setlength{\heightPublicatio}{24cm + 2\marksIndent}
	
	\newlength{\marginVerticalSuperior}
	\setlength{\marginVerticalSuperior}{1.75cm + \marksIndent}

	\newlength{\marginVerticalInferior}
	\setlength{\marginVerticalInferior}{1.75cm + \marksIndent}
	
	\newlength{\marginInterior}
	\setlength{\marginInterior}{2.5cm + \marksIndent}

	\geometry{ 
		twoside,
		vmargin = {\marginVerticalSuperior, \marginVerticalInferior}, 
		includehead, includefoot,
		left = \marginInterior,
		headheight = 0.5cm,	
		headsep = 0.75cm,
		footskip = 1.25cm,
		textwidth = 13cm, %totalheight = 20cm, % Text box 13cm x 21 cm
		paperwidth = \widthPublicatio, paperheight = \heightPublicatio,
		}

	% --------------------------------------------
	% Crop marks
	
	\ifCROP
		\usepackage[cross, a4, center, odd, noinfo]{crop}
	\fi

\fi


% ---------------------------------------------------------------------
% Headers and footers

\RequirePackage{fancyhdr}
\pagestyle{fancy}

\renewcommand{\chaptermark}[1]{\markboth{\@chapapp\ \thechapter.\ #1}{}}
\renewcommand{\sectionmark}[1]{\markright{\thesection\ #1}{}}

\fancyhead{} % Deleting current settings
\fancyfoot{} % 


\newlength{\myevenheadlinehoffset}
\newlength{\myevenheadtexthoffset}
\newlength{\myoddheadlinewidth}
\newlength{\myevenheadlinewidth}

\setlength{\myevenheadlinehoffset}{0.0cm}
\setlength{\myevenheadtexthoffset}{0.0cm}
\setlength{\myoddheadlinewidth}{\textwidth}
\setlength{\myevenheadlinewidth}{\textwidth}

\newlength{\myevenfootlinehoffset}
\newlength{\myoddfootlinehoffset}
\newlength{\myfootlinewidth}

\setlength{\myevenfootlinehoffset}{0cm}
\setlength{\myoddfootlinehoffset}{\textwidth}
\setlength{\myfootlinewidth}{1cm}


\ifAfour

    \fancypagestyle{preamb}{% 
        \fancyhead[L]{%
            \hspace*{\myevenheadlinehoffset}\makebox[0cm][l]{\rule[-2mm]{\myevenheadlinewidth}{.25pt}}%
            \hspace*{\myevenheadtexthoffset}{\small\itshape\nouppercase\leftmark}%
        } 
        \fancyfoot[LE]{%
		\makebox[0cm][l]{\rule[2.0ex]{\myfootlinewidth}{.25pt}}%
		{\makebox[\myfootlinewidth][l]{\small\thepage}}%
		}

	  \fancyfoot[RO]{%
		\hspace*{-\myfootlinewidth}\makebox[0cm][l]{\rule[2.0ex]{\myfootlinewidth}{.25pt}}%
		{\makebox[\myfootlinewidth][r]{\small\thepage}}%
		}
    }

    \fancypagestyle{myfancy}{
        \fancyhead[LE]{%
		\hspace*{\myevenheadlinehoffset}\makebox[0cm][l]{\rule[-2mm]{\myevenheadlinewidth}{.25pt}}%
		\hspace*{\myevenheadtexthoffset}{\small\itshape\nouppercase\leftmark}%
		}
	  \fancyhead[LO]{%
		\makebox[0cm][l]{\rule[-2mm]{\myoddheadlinewidth}{.25pt}}%
		\makebox[\textwidth][r]{\small\itshape\nouppercase\rightmark}%
		}

        \fancyfoot[LE]{%
		\makebox[0cm][l]{\rule[2.0ex]{\myfootlinewidth}{.25pt}}%
		{\makebox[\myfootlinewidth][l]{\small\thepage}}%
		}

	  \fancyfoot[RO]{%
		\hspace*{-\myfootlinewidth}\makebox[0cm][l]{\rule[2.0ex]{\myfootlinewidth}{.25pt}}%
		{\makebox[\myfootlinewidth][r]{\small\thepage}}%
		}
    }

    \fancypagestyle{plain}{% 
	    \fancyhf{} 
        \fancyfoot[LE]{%
		\makebox[0cm][l]{\rule[2.0ex]{\myfootlinewidth}{.25pt}}%
		{\makebox[\myfootlinewidth][l]{\small\thepage}}%
		}
		\fancyfoot[RO]{%
			\hspace*{-\myfootlinewidth}\makebox[0cm][l]{\rule[2.0ex]{\myfootlinewidth}{.25pt}}%
			{\makebox[\myfootlinewidth][r]{\small\thepage}}%
			}
		}

    \renewcommand{\headrulewidth}{0.0pt}
    \renewcommand{\footrulewidth}{0.0pt}

\else % thesis

	\fancyhead[LE]{%
		\hspace*{\myevenheadlinehoffset}\makebox[0cm][l]{\rule[-2mm]{\myevenheadlinewidth}{.25pt}}%
		\hspace*{\myevenheadtexthoffset}{\footnotesize\itshape\nouppercase\leftmark}%
		}
	\fancyhead[LO]{%
		\makebox[0cm][l]{\rule[-2mm]{\myoddheadlinewidth}{.25pt}}%
		\makebox[\textwidth][r]{\footnotesize\itshape\nouppercase\rightmark}%
		}
		
	\fancyfoot[LE]{%
		\makebox[0cm][l]{\rule[2.5ex]{\myfootlinewidth}{.25pt}}%
		{\makebox[\myfootlinewidth][l]{\thepage}}%
		}

	\fancyfoot[RO]{%
		\hspace*{-\myfootlinewidth}\makebox[0cm][l]{\rule[2.5ex]{\myfootlinewidth}{.25pt}}%
		{\makebox[\myfootlinewidth][r]{\thepage}}%
		}
		
	\fancypagestyle{plain}{% 
	    \fancyhf{} 
		\fancyfoot[RO]{%
			\hspace*{-\myfootlinewidth}\makebox[0cm][l]{\rule[2.5ex]{\myfootlinewidth}{.25pt}}%
			{\makebox[\myfootlinewidth][r]{\thepage}}%
			}
		}

	\renewcommand{\headrulewidth}{0.0pt}
	\renewcommand{\footrulewidth}{0.0pt}

\fi

% ---------------------------------------------------------------------
% Paragraph formatting and layout

\ifAfour

	\setlength{\parskip}{3ex}
	\setlength{\parindent}{0pt}

\else

	\setlength{\parskip}{2ex}
	\setlength{\parindent}{0pt}

\fi

\linespread{1.0}

\setlength{\widowpenalty}{10000pt}
\setlength{\clubpenalty}{10000pt}

\raggedbottom


% ---------------------------------------------------------------------
% Improving titles of figures and tables

\RequirePackage{caption}

\renewcommand{\captionlabelfont}{\bfseries\small}  % Bold and small for the label
\renewcommand{\captionfont}{\rmfamily\small}  % Regular and small for the caption text
%\captionsetup{labelfont=bf, textfont=it}

% ------------------------------------------------------------------------
% Section formatting

\RequirePackage[
	raggedright,
	compact,
	nobottomtitles*, 
	clearempty, 
	]{titlesec}
	
\RequirePackage{tikz}	

% ------------------------------------------------------------------------

\titleformat{\part}
	[display]
	{\thispagestyle{empty}\filcenter
	\tolerance=10000\hyphenpenalty=10000}
	{\fontsize{18}{20}\bfseries\partname\enspace\thepart}
	{1pc}
	{\fontsize{24}{30}\bfseries}

% ------------------------------------------------------------------------

%\titleformat{\chapter}
%[hang]
%{\normalfont\Large}
%%{\raggedleft\fontsize{18}{20}\bfseries\chaptertitlename\enspace\thechapter}
%{\fontsize{98}{100}\bfseries\textcolor{gray!25}{\thechapter}\hfill}
%{0ex}
%{\hfill\fontsize{24}{30}\bfseries}%\usefont{T1}{cmss}{n}{n}\selectfont
%[\vspace{1.3cm}]

\newcommand{\chapnumfont}{
	\fontsize{81}{0}
	\selectfont
}

\colorlet{chapnumcol}{black}

%\titleformat{\chapter}[display]
%{\bfseries}
%{\begin{tikzpicture}
%	\node[minimum width=\textwidth, text=black!7, fill=black!7, inner sep=1, outer sep=0, anchor=south] (rectinit) {\huge CHAPTER};
%	\node[minimum width=.8\textwidth, text=black, inner sep=1, outer sep=0, anchor=south west, text width=.8\textwidth, align=left] at (rectinit.south west) (chapname) {\Large{~~~~~}};
%	\node[minimum width=.2\textwidth, inner sep=0, outer sep=0, anchor=south west, text width=.2\textwidth, align=left] at (chapname.south east) {\chapnumfont\textcolor{chapnumcol}{~~~~~\thechapter}};
%	\end{tikzpicture}}
%{0pt}
%%{\hfill\fontsize{23}{30}\bfseries}
%{\raggedleft\fontsize{23}{30}\bfseries\textcolor{LUgray}}
%[\vspace{1cm}]

\titleformat{\chapter}{\bfseries\huge}{{\fontsize{60}{72}\selectfont\thechapter}}{1em}{}
[\vspace{1cm}]

\titlespacing*{\chapter}{0pt}{-23pt}{23pt}
% ------------------------------------------------------------------------

\ifAfour

	% ------------------------------------------------------------------------
	
	\titleformat{\section}
		[hang]
		{\vspace{1.5ex}\raggedright\tolerance=10000\hyphenpenalty=10000}
		{\Large\bfseries\thesection}
		{1em}
		{\Large\bfseries}
	
	% ------------------------------------------------------------------------
	
	\titleformat{\subsection}
		[hang]
		{\vspace{1.5ex}\raggedright\tolerance=10000\hyphenpenalty=10000}
		{\large\bfseries\itshape\thesubsection}
		{1em}
		{\large\bfseries\itshape}
		[\vspace{-1ex}]


\else % thesis


	% ------------------------------------------------------------------------
	
	\titleformat{\section}
		[hang]
		{\vspace{2ex}\raggedright\tolerance=10000\hyphenpenalty=10000}
		{\fontsize{12}{14}\bfseries\thesection}
		{1em}
		{\fontsize{12}{14}\bfseries}
	
	% ------------------------------------------------------------------------
	
	\titleformat{\subsection}
		[hang]
		{\vspace{1.5ex}\raggedright\tolerance=10000\hyphenpenalty=10000}
		{\fontsize{10.5}{12.5}\bfseries\itshape\thesubsection}
		{1em}
		{\fontsize{10.5}{12.5}\bfseries\itshape}
		[\vspace{-1ex}]
	
\fi


% ------------------------------------------------------------------------

\titleformat{\subsubsection}
	[hang]
	{\vspace{2ex}\raggedright\tolerance=10000\hyphenpenalty=10000}
	{\normalsize\itshape\thesubsubsection}
	{1em}
	{\normalsize\itshape}
	[\vspace{-0.75ex}]

% -------------------------------------------------------
% To control the numbering and format of the table of contents

\setcounter{tocdepth}{1}

\RequirePackage{titletoc}

\titlecontents{part} 
	[0em] 
	{\addvspace{4ex}\Large} 
	{\partname\enspace\thecontentslabel\enspace} 
	{} 
	{\hfill\contentspage} 
	[\vspace{-1ex}]
	
\titlecontents{chapter} 
	[0em] 
	{\addvspace{3ex}\mdseries\large} 
	{\thecontentslabel\enspace} 
	{} 
	{\hfill\contentspage}
	[\vspace{-1ex}]
	
\titlecontents{section} 
	[1.5em] 
	{\addvspace{.5ex}\small} 
	{\thecontentslabel\enspace} 
	{} 
	{\titlerule*[0.5pc]{.}\contentspage} 
	[\vspace{-1.5ex}]	
	
\titlecontents{subsection} 
	[3.5em] 
	{\vspace{.25ex}\footnotesize}
	{\thecontentslabel\enspace} 
	{} 
	{\titlerule*[0.5pc]{.}\contentspage} 
	[\vspace{-1.5ex}]


\ifAfour
\titlecontents{section} 
	[1.5em] 
	{\addvspace{0ex}\small} 
	{\thecontentslabel\enspace} 
	{} 
	{\titlerule*[0.5pc]{.}\contentspage} 
	[\vspace{-1.5ex}]	
\fi

% ------------------------------------------------------------------------
% ------------------------------------------------------------------------
% ------------------------------------------------------------------------

\RequirePackage{enumitem}
\setlist[1]{partopsep=-1ex,parsep=\parskip,itemsep=0\parskip}
\setlist[2]{partopsep=-1ex,parsep=\parskip,itemsep=0\parskip}

\makeatletter
\let\ifes@LaTeXe\iftrue
\makeatother

% ------------------------------------------------------------------------
% ------------------------------------------------------------------------
% ------------------------------------------------------------------------

\RequirePackage{mathtools}

\ifDistancesEquations
\AtBeginDocument
	{
	% ---------------------------------------------------
	% Distances of equations in the text
	
	% For normal equations
	\abovedisplayshortskip = -1.0ex plus 0ex minus 0.25ex
	\belowdisplayshortskip = 2.0ex plus 1ex minus 0.0ex
	
	% For equations in several lines
	\abovedisplayskip = -1.0ex plus 0ex minus 0.25ex
	\belowdisplayskip = 2.0ex plus 1ex minus 0.0ex
	}
\fi

% ------------------------------------------------------------------------
% ------------------------------------------------------------------------
% ------------------------------------------------------------------------
