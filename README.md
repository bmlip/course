# 5SSD0 Course Homepage, *2026-2027 edition*
### Bayesian Machine Learning and Information Processing (at [TU Eindhoven](https://www.tue.nl/en/))


> [!WARNING]
> This site is currently under construction. This is the class for 11 Nov 2026 - 28 Jan 2027 (2nd quarter).


<img src="https://github.com/bmlip/course/blob/main/assets/figures/5SSD0-banner.png?raw=true">

### Course goals
This course provides an introduction to Bayesian machine learning and information processing systems. The Bayesian approach affords a unified and consistent treatment of many useful information processing systems.

### Course summary
This course covers the fundamentals of a Bayesian (i.e., probabilistic) approach to machine learning and information processing systems. The Bayesian approach provides a unified, consistent framework for many model-based machine learning techniques. 

Initially, we focus on Linear Gaussian systems and will discuss many useful models and applications, including common regression and classification methods, Gaussian mixture models, hidden Markov models, and Kalman filters. We will discuss essential algorithms for parameter estimation in these models, including the _Variational Bayes_ method. 

The Bayesian method also provides tools for comparing the _performance_ of different information-processing systems by estimating the _Bayesian evidence_ for each model. We will discuss several methods for approximating Bayesian evidence. 

Next, we will discuss intelligent _agents_ that learn purposeful behavior through interactions with their environment. These agents are used in applications such as self-driving cars and the interactive design of virtual and augmented realities. 

Indeed, in this course, we relate synthetic Bayesian intelligent agents to natural intelligent agents such as the brain. You will be challenged to code Bayesian machine learning algorithms yourself and apply them to practical information processing problems.



<h2 style="color:red;">
News and Announcements
</h2>

<!---
- See Piazza
- (2 March 2026) See [this note on the Resit exam](https://piazza.com/class/mh9arwinped720/post/63). 
--->


## Instructors

- [Prof.dr.ir. Bert de Vries](http://bertdv.nl) (email: bert.de.vries@tue.nl) is the responsible instructor for this course and teaches the [lectures with label B](#lectures).
- [Dr. Wouter Kouw](https://biaslab.org/author/wouter-kouw/) (w.m.kouw@tue.nl) teaches the probabilistic programming [lectures with label W](#lectures).
- [Dr. Dmitry Bagaev](https://biaslab.org/author/dmitry-bagaev/) is our teaching assistant.
- [Fons van der Plas](https://biaslab.org/author/fons-van-der-plas/) is our educational advisor who is responsible for all Pluto-related issues. If you have ideas on making the course more interactive, contact Fons.


## Materials

All course materials are available in the table below. If necessary, you can download the lecture notes in PDF format here:

- [B lecture notes](https://github.com/bmlip/course/releases/download/v5.2/BMLIP.B.Lectures.pdf) version 12-Nov-2025
- [W lecture notes](https://github.com/bmlip/course/releases/download/v5.2/BMLIP.W.Lectures.pdf) version 12-Nov-2025

We recommend that you read the lecture notes in your browser to take advantage of the interactive materials we prepared for this course, which are based on [Pluto.jl](https://plutojl.org/).

### Books

The following (freely downloadable) book is optional but can be useful for additional reading:

- Christopher M.
Bishop (2006), [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf).

<!---
- [Ariel Caticha](https://www.albany.edu/physics/acaticha.shtml) (2012), [Entropic Inference and the Foundations of Physics](https://github.com/bmlip/course/blob/main/assets/files/Caticha-2012-Entropic-Inference-and-the-Foundations-of-Physics.pdf).
--->


### Software

Please follow the [software installation instructions](https://github.com/bmlip/course/blob/main/Software%20installation.md). If you encounter any problems, please get in touch with us in class or on Piazza.


### <a name="lectures">Lecture notes, assignments and video recordings</a>


You can access all lecture materials online through the links below:

<table border = "1">
         <tr>
            <th rowspan = "2"; style="text-align:center">Date</th>
            <th rowspan = "2"; style="text-align:center">lesson</th>
            <th colspan = "3"; style="text-align:center">materials</th>
         </tr>
         <tr>
            <th style="text-align:center">lecture notes</th>
            <th style="text-align:center">assignments</th>
            <th style="text-align:center">video recordings (2023/24)</th>
         </tr>
         <tr>
            <td>11-Nov-2026 <em>(Wed)</em></td>
            <td>⚪️ B0: Course Syllabus<br/>
            ⚪️ B1: Machine Learning Overview</td>
            <td><a href="https://bmlip.github.io/course/lectures/Course%20Syllabus.html">B0</a>, <a href="https://bmlip.github.io/course/lectures/Machine%20Learning%20Overview.html">B1</a></td>
            <td></td>
            <td> <a href="https://youtu.be/GtukVrtcXe8">B0</a>,  <a href="https://youtu.be/mPc3j7XgZHM">B1</a></td>
         </tr>
         <tr>
            <td>13-Nov-2026 <em>(Fri)</em></td>
            <td>⚪️ B2: Probability Theory Review</td>
            <td><a href="https://bmlip.github.io/course/lectures/Probability%20Theory%20Review.html">B2</a></td>
            <td></td>
            <td><a href="https://youtu.be/PGbN5rv6HL4">B2.1</a>, <a href="https://youtu.be/LKh2ypFVGwY">B2.2</a></td>
         </tr>
         <tr>
            <td>18-Nov-2026 <em>(Wed)</em></td>
            <td>⚪️ B3: Bayesian Machine Learning</td>
            <td><a href="https://bmlip.github.io/course/lectures/Bayesian%20Machine%20Learning.html">B3</a></td>
            <td></td>
            <td> <a href="https://youtu.be/OPGrqnnEfU0">B3.1</a>, <a href="https://youtu.be/BOUmzY1Nx5g">B3.2</a> </td>
         </tr>
         <tr>
            <td >20-Nov-2026 <em>(Fri)</em></td>
            <td >⚪️ B4: Factor Graphs and the Sum-Product Algorithm</td>
            <td><a href="https://bmlip.github.io/course/lectures/Factor%20Graphs.html">B4</a></td>
             <td></td>
             <td><a href="https://youtu.be/C2vvsf_Ts2g">B4.1</a>, <a href="https://youtu.be/HbUuYBMZOKw">B4.2</a></td>
         </tr>
         <tr>
            <td>25-Nov-2026 <em>(Wed)</em></td>
            <td>🟢 Introduction to Julia</td>
            <td><a href="https://bmlip.github.io/course/probprog/Intro%20to%20Julia.html">W0</a></td>
            <td></td>
            <td></td>
         </tr>
         <tr>
            <td>27-Nov-2026 <em>(Fri)</em></td>
            <td>🔴 Pick-up Julia programming assignment A0</td>
            <td><td><a href="https://github.com/bmlip/course/blob/main/assignments/A0%20-%20Julia%20Programming.jl">A0</a></td>
            <td></td>
         </tr>
         <tr>
            <td>27-Nov-2026 <em>(Fri)</em></td>
            <td>⚪️ B5: Continuous Data and the Gaussian Distribution</td>
            <td><a href="https://bmlip.github.io/course/lectures/The%20Gaussian%20Distribution.html">B5</a></td>
            <td></td>
            <td> <a href="https://youtu.be/WS6gWO5vgtc">B5.1</a>, <a href="https://youtu.be/Ma3jXNbNCyc">B5.2 </a> </td>
         </tr>
         <tr>
            <td>02-Dec-2026 <em>(Wed)</em> </td>
            <td>⚪️ B6: Discrete Data and the Multinomial Distribution</td>
            <td><a href="https://bmlip.github.io/course/lectures/The%20Multinomial%20Distribution.html">B6</a></td>
            <td></td>
            <td><a href="https://youtu.be/vyh8RvXxnT8">B6</a> </td>
         <tr >
            <td>04-Dec-2026 <em>(Fri)</em></td>
            <td>🟢 Probabilistic Programming 1 - Bayesian inference with conjugate models</td>
            <td><a href="https://bmlip.github.io/course/probprog/PP1%20-%20Bayesian%20inference%20in%20conjugate%20models.html">W1</a></td>
            <td></td>
            <td> <a href="https://youtu.be/ynfvgtjOnqo">W1.1</a>, <a href="https://youtu.be/h9nODl50m_M">W1.2 </a> </td>
                     </tr>
          <tr>
            <td>04-Dec-2026</td>
            <td>🔴 Submission deadline assignment A0</td>
            <td></td>
            <td><a href="">submit</a></td>
            <td></td>
         </tr>
         </tr>
         <tr>
            <td>04-Dec-2026</td>
            <td>🔴 Pick-up probabilistic programming assignment A1</td>
            <td></td>
            <td>
            <a href="https://github.com/bmlip/course/blob/main/assignments/A1%20-%20Observational%20astronomy.jl">A1</a> (<a href="https://github.com/bmlip/course/blob/main/Software%20installation.md#how-to-open-an-assignment">how to open</a>)
            </td>
            <td></td>
         </tr>
         <tr>
            <td>09-Dec-2026 <em>(Wed)</em></td>
            <td>⚪️ B7: Regression</td>
            <td><a href="https://bmlip.github.io/course/lectures/Regression.html">B7</a></td>
            <td></td>
            <td> <a href="https://youtu.be/2llpaRSN2Wc">B7.1</a>, <a href="https://youtu.be/TSoYnO6oXhw">B7.2 </a></td>
         </tr>
         <tr>
            <td>11-Dec-2026 <em>(Fri)</em></td>
            <td>⚪️ B8: Generative Classification <br/>⚪️ B9: Discriminative Classification
            </td>
            <td><a href="https://bmlip.github.io/course/lectures/Generative%20Classification.html">B8</a>, <a href="https://bmlip.github.io/course/lectures/Discriminative%20Classification.html">B9</a></td>
            <td></td>
            <td><a href="https://youtu.be/IzNDzIcrhLA">B8</a>, <a href="https://youtu.be/Y7q0JQKNfjM">B9</a></td>
         </tr>
         <tr>
            <td>16-Dec-2026 <em>(Wed)</em></td>
            <td>🟢 Probabilistic Programming 2 - Bayesian regression & classification</td>
            <td><a href="https://bmlip.github.io/course/probprog/PP2%20-%20Bayesian%20regression%20and%20classification.html">W2</a></td>
            <td></td>
            <td><a href="https://youtu.be/TKvI5uUYY8A">W2.1</a>, <a href="https://youtu.be/WCtInHz5-zA">W2.2</a></td>
         </tr>
         <tr>
            <td>18-Dec-2026 <em>(Fri)</em></td>
            <td>⚪️ B10: Latent Variable Models and Variational Bayes</td>
            <td><a href="https://bmlip.github.io/course/lectures/Latent%20Variable%20Models%20and%20VB.html">B10</a></td>
            <td></td>
            <td><a href="https://youtu.be/pVWdm9fQT6Y">B10.1</a>, <a href="https://youtu.be/mg9HGykqEbw">B10.2</a></td>
         </tr>
         <tr>
            <td>18-Dec-2026 </td>
            <td>🔴 Submission deadline assignment A1</td>
            <td></td>
            <td><a href="https://canvas.tue.nl/courses/33478/assignments/149316">submit</a>
            </td>
            <td></td>
         </tr>
         <tr>
            <td colspan="6" style="text-align:center">🔵 break</td>
         </tr>
         <tr>
            <td>06-Jan-2027 <em>(Wed)</em></td>
            <td>🟢 Probabilistic Programming 3 - Variational Bayesian inference</td>
            <td><a href="https://bmlip.github.io/course/probprog/PP3%20-%20variational%20Bayesian%20inference.html">W3</a></td>
            <td></td>
            <td><a href="https://youtu.be/z_hKaRqpNQM">W3.1</a>, <a href="https://youtu.be/FLKbzyiQlLo">W3.2</a></td>
         </tr>
         <tr>
         <tr>
            <td>08-Jan-2027 <em>(Fri)</em></td>
            <td>⚪️ B11: Dynamic Models</td>
            <td><a href="https://bmlip.github.io/course/lectures/Dynamic%20Models.html">B11</a></td>
            <td></td>
            <td><a href="https://youtu.be/W1AkZJAjvqI">B11</a></td>
         </tr>
         <tr>
            <td>08-Jan-2027</td>
            <td>🔴 Pick-up probabilistic programming assignment A2</td>
            <td></td>
            <td>
            <a href="https://github.com/bmlip/course/blob/main/assignments/A2%20-%20Tracking%20satellites.jl">A2</a> (<a href="https://github.com/bmlip/course/blob/main/Software%20installation.md#how-to-open-an-assignment">how to open</a>)
            </td>
            <td></td>
         </tr>
         <tr>
            <td>13-Jan-2027 <em>(Wed)</em></td>
            <td>⚪️ B12: Intelligent Agents and Active Inference</td>
            <td><a href="https://bmlip.github.io/course/lectures/Intelligent%20Agents%20and%20Active%20Inference.html">B12,</a><br/> <a href="https://github.com/bmlip/course/blob/main/lectures/bdv-Nov2025-AIF-lecture.ppsx">slides</a> </td>
            <td></td>
            <td><a href="https://youtu.be/fBm1oAzlv0w">B12.1</a>,  <a href="https://youtu.be/UbOuLxv9EdI">B12.2</a> </td>
         </tr>
         <tr>
            <td>15-Jan-2027 <em>(Fri)</em></td>
            <td>🟢 Probabilistic Programming 4 - Bayesian filters & smoothers</td>
            <td><a href="https://bmlip.github.io/course/probprog/PP4%20-%20Bayesian%20filtering%20and%20smoothing.html">W4</a></td>
            <td></td>
            <td><a href="https://youtu.be/Yp2vhndnjng">W4.1</a>, <a href="https://youtu.be/qnWofDRh5eo">W4.2</a></td>
         </tr>
         <tr>
            <td>22-Jan-2027 <em>(Fri)</em></td>
            <td>🔴 Submission deadline assignment A2</td>
            <td></td>
            <td><a href="https://canvas.tue.nl/courses/33478/assignments/149318">submit</a>
            </td>
            <td></td>
         </tr>
         <tr>
            <td>28-Jan-2027 (Thu)</td>
            <td colspan="5">🔵 written examination (13:30-16:30)</td>
         </tr>
         <tr>
            <td> TBD  </td>
            <td>🔴 Pick-up resit programming assignment</td>
            <td></td>
            <td><a href="https://github.com/bmlip/course/blob/main/assignments/Resit%20-%20robot%20localization.jl">Resit</a> (<a href="https://github.com/bmlip/course/blob/main/Software%20installation.md#how-to-open-an-assignment">how to open</a>)
            </td>
            <td></td>
         </tr>
         <tr>
            <td> TBD </td>
            <td>🔴 Submission deadline resit assignment</td>
            <td></td>
            <td><a href="https://canvas.tue.nl/courses/33478/assignments/155532">submit</a>
            </td>
            <td></td>
         </tr>
         <tr>
            <td > TBD </td>
            <td colspan="5">🔵 resit written examination (18:00-21:00)</td>
         </tr>
         <!-- <tr>
            <td></td>
            <td>M1: Bonus Lecture: What is Life?</td>
            <td><a href="https://youtu.be/MGusn1JzqVs">M1.1</a>, <a href="https://youtu.be/Gk225kuulOE">M1.2</a> </td>
            <td><a href="https://github.com/bertdv/BMLIP/raw/master/lessons/notebooks/MKoudahl-March2020-What-is-life.pdf">M1</a></td>
            <td></td>
         </tr> -->
      </table>

### Mini lectures
Throughout the course, you can read _Minis_ that deep-dive into specific topics. You can find the full list of minis here:

<table border = "1">
    <tr>
        <th>Mini:</th>
             <td>🟡 Sum and product of Gaussian variables</td>
        <td><a href="https://bmlip.github.io/course/minis/Sum%20and%20product%20of%20Gaussians.html">link</a></td>
    </tr>
    <tr>
        <th>Mini:</th>
             <td>🟡 Distributions in Julia</td>
        <td><a href="https://bmlip.github.io/course/minis/Distributions%20in%20Julia.html">link</a></td>
    </tr>
    <tr>
        <th>Mini:</th>
             <td>🟡 Basis Functions</td>
        <td><a href="https://bmlip.github.io/course/minis/Basis%20Functions.html">link</a></td>
    </tr>
    <tr>
        <th>Mini:</th>
             <td>🟡 The Softmax Function</td>
        <td><a href="https://bmlip.github.io/course/minis/Softmax.html">link</a></td>
    </tr>
    <tr>
        <th>Mini:</th>
             <td>🟡 Generative Classification</td>
        <td><a href="https://bmlip.github.io/course/minis/Generative%20Classification.html">link</a></td>
    </tr>
    <tr>
        <th>Mini:</th>
             <td>🟡 Laplace Approximation</td>
        <td><a href="https://bmlip.github.io/course/minis/Laplace%20Approximation.html">link</a></td>
    </tr>
    <tr>
        <th>Mini:</th>
             <td>🟡 Kullback-Leibler Divergence</td>
        <td><a href="https://bmlip.github.io/course/minis/KL%20Divergence.html">link</a></td>
    </tr>
    <tr>
        <th>Mini:</th>
             <td>🟡 How to use this knowledge</td>
        <td><a href="https://bmlip.github.io/course/minis/How%20to%20use%20this%20knowledge.html">link</a></td>
    </tr>
    <tr>
        <th>Mini:</th>
             <td>🟡 RxInfer Tips & Tricks</td>
        <td><a href="https://bmlip.github.io/course/minis/RxInfer%20tips%20and%20tricks.html">link</a></td>
    </tr>
</table>

## Exams & Assignments

### Exam Rules

- You can not bring a formula sheet, nor use a phone or calculator at the exam. This [Formula Sheet](https://github.com/bmlip/course/blob/main/assets/files/5SSD0_formula_sheet.pdf) will be provided in the preamble of the exam. You can use the formula sheet when making any exercises. 


### Exam Preparation

- The written exam will be a multiple-choice exam, just like the examples below. This year, there will be no probabilistic programming question in the written exam.

- In addition to the materials in the above table, we provide two representative practice written exams:

  - 3-Feb-2022: <a href="https://github.com/bmlip/course/blob/main/exams/20220203-5SSD0-exam.pdf">exam </a>; <a href="https://github.com/bmlip/course/blob/main/exams/20220203-5SSD0-exam-answers.pdf">answers </a>; <a href="https://github.com/bmlip/course/blob/main/exams/20220203-5SSD0-exam-calculations.ipynb">calculations</a>
  - 2-Feb-2023: <a href="https://github.com/bmlip/course/blob/main/exams/20230202-5SSD0-exam.pdf">exam </a>; <a href="https://github.com/bmlip/course/blob/main/exams/20230202-5SSD0-exam-answers.pdf">answers </a>; <a href="https://github.com/bmlip/course/blob/main/exams/20230202-5SSD0-exam-calculations.ipynb">calculations</a>


### Programming Assignments

- Programming assignments can be downloaded and submitted through the links in the above table.

<!---
- Programming assignments should be submitted before the indicated deadlines at the [Canvas Assignments tab](https://canvas.tue.nl/courses/26086/assignments).
--->

### Grading

- The final grade is composed of the results of assignments A1 (10%),  A2 (10%), and the final written exam (80%). The grade will be rounded to the nearest integer.


<!---
## Projects

- If you liked this class, [here is a short oversight](https://youtu.be/G7EJqWY4aq0) (~10 minutes) of internship and graduation projects that you may consider applying for.
--->

<!---
### Oral Exam Guide

An exam session lasts about 30 minutes and will be recorded (and later deleted, following GDPR rules). At the beginning of the session, the examiner needs to check your identity, preferably by your campus card.

The style of the examination is conversational. We like to engage in a conversation with you about what you learned in the class. In general, oral exams do not lend themselves well to proofing theorems or other deep mathematical manipulations. Instead, the focus is more on testing if you understand the conceptual ideas in this class. In principle, everything that has been presented in the lecture notes and videos is fair game as an exam question, including programming questions from the probabilistic programming sessions.

Please review the [Oral Exam Example notebook](https://nbviewer.jupyter.org/github/bertdv/BMLIP/blob/master/lessons/exercises/Oral-Exam-Example.ipynb (Links to an external site.)) to get an idea of what kind of questions will be asked.

The first question of the exam will be an open question: "You get 5 minutes to tell me about what you learned in this class. You can fill in the 5 minutes as you like but try to impress me with your knowledge or insights. E.g., talk about probabilistic modelling, how it works, what are strong aspects or weak aspects of the approach, etc." After the first question, the rest of the exam will be focused at topics selected by the examiner.
--->




<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>






<!---
# Behind the scenes
For instructors:

> [!IMPORTANT]
> The Pluto notebooks in this repository (`.jl` files) are automatically rendered on our website. You can view them online at https://bmlip.github.io/course/, and copy URLs from this index to use in the course schedule.
>
> *Status for live interactivity (PlutoSliderServer):* [![Better Stack Badge](https://uptime.betterstack.com/status-badges/v1/monitor/1svzl.svg)](https://tue-bmlip.betteruptime.com/)
--->

## How to modify lecture materials

Take a look at https://github.com/bmlip/course/tree/main/developer%20instructions for more information aimed at the course lecturers and website admins.

