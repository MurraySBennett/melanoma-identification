// Define study
const study = lab.util.fromObject({
  "title": "root",
  "type": "lab.flow.Sequence",
  "parameters": {},
  "plugins": [
    {
      "type": "lab.plugins.Metadata",
      "path": undefined
    },
    {
      "type": "global.Pavlovia",
      "path": undefined
    }
  ],
  "metadata": {
    "title": "BTL_Melanoma",
    "description": "",
    "repository": "",
    "contributors": ""
  },
  "files": {},
  "responses": {},
  "content": [
    {
      "type": "lab.html.Screen",
      "files": {},
      "responses": {
        "click button": "continue",
        "keypress": "continue"
      },
      "parameters": {},
      "messageHandlers": {
        "run": function anonymous(
) {
let conditions = ["irregular", "consistent", "colourful", "uniform"];
this.parameters.condition = this.random.choice(conditions)
console.log(this.parameters.condition)

if (this.parameters.condition === conditions[0]){
  var condition_instruction1 = "You will be presented with two images of skin lesions throughout this experiment. Your task is to select which of the two lesions has the <strong>more irregular border</strong>."
  var condition_instruction2 = "Images will be presented side-by-side on each trial. Please select the lesion with the <strong>more irregular border</strong> by pressing:"
  var condition_instruction3 = "You will now be presented with a brief set of practice trials. The initial trials will use images where the irregularity between borders is high, followed by some trials where irregularity is low. However, throughout the experiment, please keep in mind that, in some cases, there is no <em>right</em> answer."
  window.foot_instruction = "Which lesion has the <strong>more irregular border</strong>?<br>Left &nbsp; &nbsp; &nbsp; Right<br><kbd><strong>Z</strong></kbd>&nbsp; &nbsp; &nbsp;<kbd><strong>M</strong></kbd>";
} else if (this.parameters.condition === conditions[1]){
  var condition_instruction1 = "You will be presented with two images of skin lesions throughout this experiment. Your task is to select which of the two lesions has the <strong>more consistent border</strong>."
  var condition_instruction2 = "Images will be presented side-by-side on each trial. Please select the lesion with the <strong>more consistent border</strong> by pressing:"
  var condition_instruction3 = "You will now be presented with a brief set of practice trials. The initial trials will use images where the consistency between borders is high, followed by some trials where border consistency is low. However, throughout the experiment, please keep in mind that, in some cases, there is no <em>right</em> answer."
  window.foot_instruction = "Which lesion has the <strong>more consistent border</strong>?<br>Left &nbsp; &nbsp; &nbsp; Right<br><kbd><strong>Z</strong></kbd>&nbsp; &nbsp; &nbsp;<kbd><strong>M</strong></kbd>";
} else if (this.parameters.condition == conditions[2]){
  var condition_instruction1 = "You will be presented with two images of skin lesions throughout this experiment. Your task is to select which of the two lesions has the <strong>more colour variation</strong>."
  var condition_instruction2 = "Images will be presented side-by-side on each trial. Please select the lesion with the <strong>more colour variation</strong> by pressing:"
  var condition_instruction3 = "You will now be presented with a brief set of practice trials. The initial trials will use images where the difference in colour variation between lesions is high, followed by some trials where colour variation is similar. However, throughout the experiment, please keep in mind that, in some cases, there is no <em>right</em> answer."
  window.foot_instruction = "Which lesion has <strong>more colour variation</strong>?<br>Left &nbsp; &nbsp; &nbsp; Right<br><kbd><strong>Z</strong></kbd>&nbsp; &nbsp; &nbsp;<kbd><strong>M</strong></kbd>";
} else if (this.parameters.condition == conditions[3]){
  var condition_instruction1 = "You will be presented with two images of skin lesions throughout this experiment. Your task is to select which of the two lesions has the <strong>more uniform colour</strong>."
  var condition_instruction2 = "Images will be presented side-by-side on each trial. Please select the lesion with the <strong>more uniform colour</strong> by pressing:"
  var condition_instruction3 = "You will now be presented with a brief set of practice trials. The initial trials will use images where the difference in colour uniformity is high, followed by some trials where colour uniformity is similar. However, throughout the experiment, please keep in mind that, in some cases, there is no <em>right</em> answer."
  window.foot_instruction = "Which lesion has the <strong>more uniform colour</strong>?<br>Left &nbsp; &nbsp; &nbsp; Right<br><kbd><strong>Z</strong></kbd>&nbsp; &nbsp; &nbsp;<kbd><strong>M</strong></kbd>";
}


document.getElementById('condition_instruction1').innerHTML = condition_instruction1;
document.getElementById('condition_instruction2').innerHTML = condition_instruction2;
document.getElementById('condition_instruction3').innerHTML = condition_instruction3;



},
        "before:prepare": function anonymous(
) {
window.practice_stimuli = [["0.JPG","1.JPG"], ["2.JPG","3.JPG"], ["4.JPG","5.JPG"], ["6.JPG","7.JPG"]];

window.img_root = "https://dok5xvq3u1gii.cloudfront.net/ISIC-database/";
window.stimuli = ["ISIC_0072175.JPG", "ISIC_0072182.JPG", "ISIC_0072194.JPG"];
}
      },
      "title": "Welcome",
      "content": "\u003Cheader\u003E\r\n  \u003Ch1 class=\"content-vertical-top content-horizontal-center\"\u003EWelcome\u003C\u002Fh1\u003E\r\n\u003C\u002Fheader\u003E\r\n\u003Cmain class=\"content-vertical-top content-horizontal-center content-vertical-space-between\"\u003E\r\n  \u003Cdiv class=\"w-l text-center alert\"\u003E\r\n    \u003Cp\u003EPlease ensure that your browser window is maximised.\u003C\u002Fp\u003E\r\n  \u003C\u002Fdiv\u003E\r\n  \u003Cdiv class=\"w-l text-justify\"\u003E\r\n    \u003Ctable\u003E\r\n    \u003Ctr id=\"intro\"\u003E\r\n      \u003Ctd\u003E \u003C\u002Ftd\u003E\r\n    \u003C\u002Ftr\u003E\r\n    \u003Ctr id=\"condition_instruction1\"\u003E\r\n      \u003Ctd\u003E  \u003C\u002Ftd\u003E\r\n    \u003C\u002Ftr\u003E\r\n    \u003Ctr id=\"condition_instruction2\"\u003E\r\n      \u003Ctd\u003E \u003C\u002Ftd\u003E\r\n    \u003C\u002Ftr\u003E\r\n    \u003Ctr id=\"response_keys\"\u003E \r\n      \u003Ctd class=\"w-l text-center\"\u003E\r\n        \u003Ckbd\u003E\u003Cstrong\u003EZ\u003C\u002Fstrong\u003E\u003C\u002Fkbd\u003E for the left image\u003Cbr\u003E\r\n        \u003Ckbd\u003E\u003Cstrong\u003EM\u003C\u002Fstrong\u003E\u003C\u002Fkbd\u003E for the right image\r\n      \u003C\u002Ftd\u003E\r\n    \u003C\u002Ftr\u003E\r\n    \u003C\u002Ftable\u003E\r\n    \r\n    \u003Ctable\u003E\r\n      \u003Ctr id=\"condition_instruction3\"\u003E \r\n        \u003Ctd\u003E \u003C\u002Ftd\u003E\r\n      \u003C\u002Ftr\u003E\r\n    \u003C\u002Ftable\u003E\r\n\r\n    \u003Cp\u003EPlease click the \u003Ckbd\u003EContinue &#8594;\u003C\u002Fkbd\u003E button below to begin.\u003C\u002Fp\u003E\r\n    \u003Cp class=\"content-vertical-top content-horizontal-center\"\u003E\r\n      \u003Cbutton id=\"continue\"\u003EContinue &rarr;\u003C\u002Fbutton\u003E\r\n    \u003C\u002Fp\u003E\r\n  \u003C\u002Fdiv\u003E\r\n\r\n\u003C\u002Fmain\u003E\r\n\r\n\u003C!-- \u003Cfooter class=\"content-horizontal-right\"\u003E\r\n  \u003Cbutton id=\"continue\"\u003EContinue &rarr;\u003C\u002Fbutton\u003E\r\n\u003C\u002Ffooter\u003E --\u003E"
    },
    {
      "type": "lab.flow.Loop",
      "templateParameters": [
        {
          "": "0"
        }
      ],
      "sample": {
        "mode": "sequential",
        "n": "${this.parameters.total_trials}"
      },
      "files": {},
      "responses": {
        "": ""
      },
      "parameters": {
        "total_trials": 1,
        "practice": true,
        "total_blocks": 1
      },
      "messageHandlers": {},
      "title": "loop",
      "shuffleGroups": [],
      "template": {
        "type": "lab.flow.Sequence",
        "files": {},
        "responses": {
          "": ""
        },
        "parameters": {},
        "messageHandlers": {},
        "title": "sequence",
        "content": [
          {
            "type": "lab.html.Screen",
            "files": {},
            "responses": {
              "": ""
            },
            "parameters": {},
            "messageHandlers": {
              "run": function anonymous(
) {
// footer 
document.getElementById('foot_instruction').innerHTML = window.foot_instruction;
}
            },
            "title": "interval",
            "content": "\u003Cmain class=\"content-horizontal-space-around content-vertical-center\"\u003E\r\n  \u003Cdiv\u003E\r\n    \u003C!-- \u003Cstrong\u003E+\u003Cstrong\u003E --\u003E\r\n  \u003C\u002Fdiv\u003E\r\n\u003C\u002Fmain\u003E\r\n\r\n\r\n\u003Cfooter class=\"content-horizontal-center content-vertical-center\"\u003E\r\n  \u003Cp id=\"foot_instruction\"\u003E \u003C\u002Fp\u003E\r\n\u003C\u002Ffooter\u003E",
            "timeout": "500"
          },
          {
            "type": "lab.html.Screen",
            "files": {},
            "responses": {
              "": ""
            },
            "parameters": {},
            "messageHandlers": {
              "run": function anonymous(
) {
// footer 
document.getElementById('foot_instruction').innerHTML = window.foot_instruction;
}
            },
            "title": "cue",
            "content": "\u003Cmain \u003Cmain class=\"content-horizontal-center content-vertical-center\"\u003E\r\n  \u003Cdiv\u003E\r\n    \u003Cp style=\"font-size: 80px\"\u003E\r\n    +\r\n    \u003C\u002Fp\u003E\r\n  \u003C\u002Fdiv\u003E\r\n\u003C\u002Fmain\u003E\r\n\r\n\u003Cfooter class=\"content-horizontal-center content-vertical-center\"\u003E\r\n  \u003Cp id=\"foot_instruction\"\u003E \u003C\u002Fp\u003E\r\n\u003C\u002Ffooter\u003E",
            "timeout": "200"
          },
          {
            "type": "lab.html.Screen",
            "files": {
              "8.JPG": "embedded\u002Fc9902cebdc103a8ae928bac780552d4acc7eb69280cda897b48dcae717bca341.JPG",
              "0.JPG": "embedded\u002F829cb62e8ae0d628100407f2a2a50e98ed4b65155e8f1f6d345ce1fcf0b4115a.JPG",
              "1.JPG": "embedded\u002F8d3cae9f08e28e88748d7a0fde8b7b9585fc9cf2992fab369f65c79c01933754.JPG",
              "2.JPG": "embedded\u002F8a51d2c36fea65fa95fc8130bd930c886d9709076c8e85c37465a06fa3767e2d.JPG",
              "3.JPG": "embedded\u002F71eae3775faecb0f1570bfb305c665bd513378bfe47c451e43ed97fce882586d.JPG",
              "4.JPG": "embedded\u002Fa2bf9b45afdcdb2ee5aff6df7e400b5ca275760e0f6275dc51fb9e008ee71cfb.JPG",
              "5.JPG": "embedded\u002F587b84f7a22bb4383a52bd309c36f06c0093c7440ce6952cb7fb9009bf628b74.JPG",
              "6.JPG": "embedded\u002F342c5f996b8b413e7c68dfd9a71776d32688492323fa397260dd3512922fdd90.JPG",
              "7.JPG": "embedded\u002F944eda56cab9031b0208bb06994bba29a703a25013e56755fac051debe418a1d.JPG"
            },
            "responses": {
              "keypress(z)": "0",
              "keypress(m)": "1",
              "keypress(Z)": "0",
              "undefined(M)": "1"
            },
            "parameters": {
              "selection_duration": "500"
            },
            "messageHandlers": {
              "run": function anonymous(
) {
window.trials_per_block = this.parameters.total_trials / this.parameters.total_blocks;

if ( ! window.blockNo)
  window.blockNo = 0;

if ( ! window.trialNo)
  window.trialNo = 1;
else
  window.trialNo++;

this.parameters.blockNo = window.blockNo;
this.parameters.trialNo = window.trialNo;

if (window.trialNo === window.trials_per_block) {
  window.trialNo = 0;
  window.blockNo++;
  window.takeBreak = true;
  // if (window.blockNo === window.total_blocks + 1 && !this.parameters.practice){
  //   window.endExperiment = true;
  // }
}
else {
  window.takeBreak = false;
  window.endExperiment = false;
}


if (this.parameters.practice){
  this.parameters.img_left = window.practice_stimuli[window.trialNo][0];
  this.parameters.img_right = window.practice_stimuli[window.trialNo][1];
} else {
  this.parameters.img_left = this.random.choice(window.stimuli);
  this.parameters.img_right = this.random.choice(window.stimuli)
}

window.new_stim = [`${ this.files[this.parameters.img_left] }`, `${ this.files[this.parameters.img_right] }`];

function changeImage(element, path) {
  element.src=path;
}

// For image selection feedback
let display_left = document.getElementById('display_left');
let display_right= document.getElementById('display_right');


let selections = {
  '0': display_left,
  '1': display_right,
}

let trial_stimuli = [this.parameters.img_left, this.parameters.img_right];

// allow a moment for image load time, thus 'ensuring' perception of equal presentation times
let load_time = 100 //ms
let load_timer = load_time;
let fps = 60;
// let last_update = 0;
// let time_since_last = last_update;
only_once = 1;
one_change = 1;

show_stim = () => {
  if (load_timer != load_time && one_change===1){
    changeImage(display_left, window.new_stim[0]);
    changeImage(display_right, window.new_stim[1]);
  }


  if (load_timer <= 0 && only_once===1){
    display_left.className += ' shown';
    display_right.className += ' shown'
    only_once = 0;
    this.parameters.img_reveal = this.parameters.timestamp;
  }
  load_timer -= 1/fps * 1000; // a better man would calculate time since last frame and substract that.
}

// footer 
document.getElementById('foot_instruction').innerHTML = window.foot_instruction;

// override respond()
this._respond = this.respond;
this.respond = (response=null, timestamp=undefined) => {
  // response === this.parameters.response;
  // this.parameters.correct = (response === this.parameters.correctResponse);
  let wait = this.parameters.selection_duration;
  selections[response].className += ' border';
  // if (this.parameters.practice) {
  //     wait = this.parameters.practiceFeedbackDuration;
  //     selections[this.parameters.correctResponse].className += ' shown';
  // }
  this.parameters.winner = trial_stimuli[response];
  this.parameters.loser  = trial_stimuli[Math.abs(response - 1)]
  
  setTimeout(() => {
    this._respond(response, timestamp);
  }, wait);
}

let iid = setInterval(show_stim, 1/fps*1000);
let end = (reason) => {
  clearInterval(iid)
  return this.originalEnd(reason)
}
this.originalEnd = this.end
this.end = end

}
            },
            "title": "trial",
            "content": "\u003C!-- CSS for these divs and imgs are managed via the settings button in the top left corner of labjs --\u003E\r\n\u003Cmain class=\"content-horizontal-center content-vertical-center\"\u003E\r\n  \u003Cdiv id=\"container\"\u003E\r\n    \u003C!-- \u003Cimg src=${ this.files[this.parameters.img_left]} id=\"display_left\" class=\"display\"\u002F\u003E   --\u003E\r\n    \u003Cimg id=\"display_left\"  class=\"display\" \u002F\u003E\r\n    \u003Cimg id=\"display_right\" class=\"display\" \u002F\u003E\r\n  \u003C\u002Fdiv\u003E\r\n\u003C\u002Fmain\u003E\r\n\r\n\u003Cfooter class=\"content-horizontal-center content-vertical-center\"\u003E\r\n  \u003Cp id=\"foot_instruction\"\u003E \u003C\u002Fp\u003E \r\n\u003C\u002Ffooter\u003E",
            "timeout": "10000"
          }
        ]
      }
    },
    {
      "type": "lab.html.Screen",
      "files": {},
      "responses": {
        "click button": "continue",
        "keypress": "continue"
      },
      "parameters": {},
      "messageHandlers": {
        "run": function anonymous(
) {
// let conditions = ["irregular", "consistent", "colourful", "uniform"];
// this.parameters.condition = this.random.choice(conditions)
// console.log(this.parameters.condition)



// if (this.parameters.condition === conditions[0]){
//   var condition_instruction1 = "You will be presented with two images of skin lesions throughout this experiment. Your task is to select which of the two lesions has the <strong>more irregular border</strong>."
//   var condition_instruction2 = "Images will be presented side-by-side on each trial. Please select the lesion with the <strong>more irregular border</strong> by pressing:"
//   var condition_instruction3 = "You will now be presented with a brief set of practice trials. The initial trials will use images where the irregularity between borders is high, followed by some trials where irregularity is low. However, throughout the experiment, please keep in mind that, in some cases, there is no <em>right</em> answer."
//   window.foot_instruction = "Which lesion has the <strong>more irregular border</strong>?<br>Left &nbsp; &nbsp; &nbsp; Right<br><kbd><strong>Z</strong></kbd>&nbsp; &nbsp; &nbsp;<kbd><strong>M</strong></kbd>";
// } else if (this.parameters.condition === conditions[1]){
//   var condition_instruction1 = "You will be presented with two images of skin lesions throughout this experiment. Your task is to select which of the two lesions has the <strong>more consistent border</strong>."
//   var condition_instruction2 = "Images will be presented side-by-side on each trial. Please select the lesion with the <strong>more consistent border</strong> by pressing:"
//   var condition_instruction3 = "You will now be presented with a brief set of practice trials. The initial trials will use images where the consistency between borders is high, followed by some trials where border consistency is low. However, throughout the experiment, please keep in mind that, in some cases, there is no <em>right</em> answer."
//   window.foot_instruction = "Which lesion has the <strong>more consistent border</strong>?<br>Left &nbsp; &nbsp; &nbsp; Right<br><kbd><strong>Z</strong></kbd>&nbsp; &nbsp; &nbsp;<kbd><strong>M</strong></kbd>";
// } else if (this.parameters.condition == conditions[2]){
//   var condition_instruction1 = "You will be presented with two images of skin lesions throughout this experiment. Your task is to select which of the two lesions has the <strong>more colour variation</strong>."
//   var condition_instruction2 = "Images will be presented side-by-side on each trial. Please select the lesion with the <strong>more colour variation</strong> by pressing:"
//   var condition_instruction3 = "You will now be presented with a brief set of practice trials. The initial trials will use images where the difference in colour variation between lesions is high, followed by some trials where colour variation is similar. However, throughout the experiment, please keep in mind that, in some cases, there is no <em>right</em> answer."
//   window.foot_instruction = "Which lesion has <strong>more colour variation</strong>?<br>Left &nbsp; &nbsp; &nbsp; Right<br><kbd><strong>Z</strong></kbd>&nbsp; &nbsp; &nbsp;<kbd><strong>M</strong></kbd>";
// } else if (this.parameters.condition == conditions[3]){
//   var condition_instruction1 = "You will be presented with two images of skin lesions throughout this experiment. Your task is to select which of the two lesions has the <strong>more uniform colour</strong>."
//   var condition_instruction2 = "Images will be presented side-by-side on each trial. Please select the lesion with the <strong>more uniform colour</strong> by pressing:"
//   var condition_instruction3 = "You will now be presented with a brief set of practice trials. The initial trials will use images where the difference in colour uniformity is high, followed by some trials where colour uniformity is similar. However, throughout the experiment, please keep in mind that, in some cases, there is no <em>right</em> answer."
//   window.foot_instruction = "Which lesion has the <strong>more uniform colour</strong>?<br>Left &nbsp; &nbsp; &nbsp; Right<br><kbd><strong>Z</strong></kbd>&nbsp; &nbsp; &nbsp;<kbd><strong>M</strong></kbd>";
// }


// document.getElementById('condition_instruction1').innerHTML = condition_instruction1;
// document.getElementById('condition_instruction2').innerHTML = condition_instruction2;
// document.getElementById('condition_instruction3').innerHTML = condition_instruction3;



}
      },
      "title": "exp_instructions",
      "content": "\u003Cheader\u003E\r\n  \u003Ch1\u003EEnd of Practice Block\u003C\u002Fh1\u003E\r\n\u003C\u002Fheader\u003E\r\n\u003Cmain class=\"content-vertical-top content-horizontal-center content-vertical-space-between\"\u003E\r\n  \u003Cdiv class=\"w-m text-center\"\u003E\r\n    \u003Cp\u003EYou are now entering the main phase of the experiment. Please ensure that your browser window is maximised.\u003C\u002Fp\u003E\r\n  \u003C\u002Fdiv\u003E\r\n  \u003Cp\u003EPress spacebar to begin.\u003C\u002Fp\u003E\r\n\u003C\u002Fmain\u003E\r\n\r\n\u003Cfooter class=\"content-horizontal-center content-vertical-center\"\u003E\r\n  \u003Cp id=\"foot_instruction\"\u003E \u003C\u002Fp\u003E \r\n\u003C\u002Ffooter\u003E"
    },
    {
      "type": "lab.flow.Loop",
      "templateParameters": [
        {
          "": "0"
        }
      ],
      "sample": {
        "mode": "sequential",
        "n": "${this.parameters.total_trials}"
      },
      "files": {},
      "responses": {
        "": ""
      },
      "parameters": {
        "total_trials": 6,
        "practice": false,
        "total_blocks": 2
      },
      "messageHandlers": {},
      "title": "loop",
      "shuffleGroups": [],
      "template": {
        "type": "lab.flow.Sequence",
        "files": {},
        "responses": {
          "": ""
        },
        "parameters": {},
        "messageHandlers": {},
        "title": "sequence",
        "content": [
          {
            "type": "lab.html.Screen",
            "files": {},
            "responses": {
              "": ""
            },
            "parameters": {},
            "messageHandlers": {
              "run": function anonymous(
) {
// footer 
document.getElementById('foot_instruction').innerHTML = window.foot_instruction;
}
            },
            "title": "interval",
            "content": "\u003Cmain class=\"content-horizontal-space-around content-vertical-center\"\u003E\r\n  \u003Cdiv\u003E\r\n    \u003C!-- \u003Cstrong\u003E+\u003Cstrong\u003E --\u003E\r\n  \u003C\u002Fdiv\u003E\r\n\u003C\u002Fmain\u003E\r\n\r\n\r\n\u003Cfooter class=\"content-horizontal-center content-vertical-center\"\u003E\r\n  \u003Cp id=\"foot_instruction\"\u003E \u003C\u002Fp\u003E\r\n\u003C\u002Ffooter\u003E",
            "timeout": "500"
          },
          {
            "type": "lab.html.Screen",
            "files": {},
            "responses": {
              "": ""
            },
            "parameters": {},
            "messageHandlers": {
              "run": function anonymous(
) {
// footer 
document.getElementById('foot_instruction').innerHTML = window.foot_instruction;
}
            },
            "title": "cue",
            "content": "\u003Cmain \u003Cmain class=\"content-horizontal-center content-vertical-center\"\u003E\r\n  \u003Cdiv\u003E\r\n    \u003Cp style=\"font-size: 80px\"\u003E\r\n    +\r\n    \u003C\u002Fp\u003E\r\n  \u003C\u002Fdiv\u003E\r\n\u003C\u002Fmain\u003E\r\n\r\n\u003Cfooter class=\"content-horizontal-center content-vertical-center\"\u003E\r\n  \u003Cp id=\"foot_instruction\"\u003E \u003C\u002Fp\u003E\r\n\u003C\u002Ffooter\u003E",
            "timeout": "200"
          },
          {
            "type": "lab.html.Screen",
            "files": {
              "8.JPG": "embedded\u002Fc9902cebdc103a8ae928bac780552d4acc7eb69280cda897b48dcae717bca341.JPG",
              "0.JPG": "embedded\u002F829cb62e8ae0d628100407f2a2a50e98ed4b65155e8f1f6d345ce1fcf0b4115a.JPG",
              "1.JPG": "embedded\u002F8d3cae9f08e28e88748d7a0fde8b7b9585fc9cf2992fab369f65c79c01933754.JPG",
              "2.JPG": "embedded\u002F8a51d2c36fea65fa95fc8130bd930c886d9709076c8e85c37465a06fa3767e2d.JPG",
              "3.JPG": "embedded\u002F71eae3775faecb0f1570bfb305c665bd513378bfe47c451e43ed97fce882586d.JPG",
              "4.JPG": "embedded\u002Fa2bf9b45afdcdb2ee5aff6df7e400b5ca275760e0f6275dc51fb9e008ee71cfb.JPG",
              "5.JPG": "embedded\u002F587b84f7a22bb4383a52bd309c36f06c0093c7440ce6952cb7fb9009bf628b74.JPG",
              "6.JPG": "embedded\u002F342c5f996b8b413e7c68dfd9a71776d32688492323fa397260dd3512922fdd90.JPG",
              "7.JPG": "embedded\u002F944eda56cab9031b0208bb06994bba29a703a25013e56755fac051debe418a1d.JPG"
            },
            "responses": {
              "keypress(z)": "0",
              "keypress(m)": "1",
              "keypress(Z)": "0",
              "undefined(M)": "1"
            },
            "parameters": {
              "selection_duration": "500"
            },
            "messageHandlers": {
              "run": function anonymous(
) {
window.trials_per_block = this.parameters.total_trials / this.parameters.total_blocks;

if ( ! window.blockNo)
  window.blockNo = 1;

if ( ! window.trialNo)
  window.trialNo = 1;
else
  window.trialNo++;

this.parameters.blockNo = window.blockNo;
this.parameters.trialNo = window.trialNo;
window.total_blocks = this.parameters.total_blocks;

if (window.trialNo === window.trials_per_block) {
  window.trialNo = 0;
  window.blockNo++;
  window.takeBreak = true;
  if (window.blockNo === window.total_blocks + 1 && !this.parameters.practice){
    window.endExperiment = true;
  }
}
else {
  window.takeBreak = false;
  window.endExperiment = false;
}

if (this.parameters.practice){
  this.parameters.img_left = window.practice_stimuli[window.trialNo][0];
  this.parameters.img_right = window.practice_stimuli[window.trialNo][1];
} else {
  // copy the stimulus list so you can remove the first random choice, 
  // thereby ensuring you don't pick doubles.
  let tmp_list = window.stimuli.map((x) => x);
  this.parameters.img_left = this.random.choice(tmp_list);
  tmp_list.splice(tmp_list.indexOf(this.parameters.img_left), 1);
  this.parameters.img_right=this.random.choice(tmp_list);
}

// window.new_stim = [`${ this.files[this.parameters.img_left] }`, `${ this.files[this.parameters.img_right] }`];
window.new_stim = [window.img_root + this.parameters.img_left, window.img_root + this.parameters.img_right];

function changeImage(element, path) {
  element.src = path;
}

// For image selection feedback
let display_left = document.getElementById('display_left');
let display_right= document.getElementById('display_right');


let selections = {
  '0': display_left,
  '1': display_right,
}

let trial_stimuli = [this.parameters.img_left, this.parameters.img_right];
console.log(window.img_root + trial_stimuli[0])

// allow a moment for image load time, thus 'ensuring' perception of equal presentation times
let load_time = 100 //ms
let load_timer = load_time;
let fps = 60;

only_once = 1;
one_change = 1;

show_stim = () => {
  if (load_timer != load_time && one_change===1){
    changeImage(display_left, window.new_stim[0]);
    changeImage(display_right, window.new_stim[1]);
  }


  if (load_timer <= 0 && only_once===1){
    display_left.className += ' shown';
    display_right.className += ' shown'
    only_once = 0;
    this.parameters.img_reveal = this.parameters.timestamp;
  }
  load_timer -= 1/fps * 1000;
}

// footer 
document.getElementById('foot_instruction').innerHTML = window.foot_instruction;

// override respond()
this._respond = this.respond;
this.respond = (response=null, timestamp=undefined) => {
  // response === this.parameters.response;
  // this.parameters.correct = (response === this.parameters.correctResponse);
  let wait = this.parameters.selection_duration;
  selections[response].className += ' border';
  // if (this.parameters.practice) {
  //     wait = this.parameters.practiceFeedbackDuration;
  //     selections[this.parameters.correctResponse].className += ' shown';
  // }
  this.parameters.winner = trial_stimuli[response];
  this.parameters.loser  = trial_stimuli[Math.abs(response - 1)]
  
  setTimeout(() => {
    this._respond(response, timestamp);
  }, wait);
}

let iid = setInterval(show_stim, 1/fps*1000);
let end = (reason) => {
  clearInterval(iid)
  return this.originalEnd(reason)
}
this.originalEnd = this.end
this.end = end
}
            },
            "title": "trial",
            "content": "\u003C!-- CSS for these divs and imgs are managed via the settings button in the top left corner of labjs --\u003E\r\n\u003Cmain class=\"content-horizontal-center content-vertical-center\"\u003E\r\n  \u003Cdiv id=\"container\"\u003E\r\n    \u003C!-- \u003Cimg src=${ this.files[this.parameters.img_left]} id=\"display_left\" class=\"display\"\u002F\u003E   --\u003E\r\n    \u003Cimg id=\"display_left\"  class=\"display\"\u002F\u003E\r\n    \u003Cimg id=\"display_right\" class=\"display\" \u002F\u003E\r\n  \u003C\u002Fdiv\u003E\r\n\u003C\u002Fmain\u003E\r\n\r\n\u003Cfooter class=\"content-horizontal-center content-vertical-center\"\u003E\r\n  \u003Cp id=\"foot_instruction\"\u003E \u003C\u002Fp\u003E \r\n\u003C\u002Ffooter\u003E",
            "timeout": "10000"
          },
          {
            "type": "lab.html.Screen",
            "files": {},
            "responses": {
              "": "any key"
            },
            "parameters": {},
            "messageHandlers": {
              "run": function anonymous(
) {
if ( ! window.takeBreak) {
  this.end();
} else if (window.endExperiment){
  this.end();
}
let block_progress = "You have completed " + String(window.blockNo - 1) + " of " + String(window.total_blocks) + " blocks.";
document.getElementById('block_progress').innerHTML = block_progress;

// const button = document.querySelector("#restart-button");
// const bars = document.querySelectorAll(".round-time-bar");
// button.addEventListener("click", () => {
//   bars.forEach((bar) => {
//     bar.classList.remove("round-time-bar");
//     bar.offsetWidth;
//     bar.classList.add("round-time-bar");
//   });
// });

}
            },
            "title": "break",
            "content": "\u003Cmain class=\"content-horizontal-center content-vertical-center\"\u003E \r\n  \u003Cdiv\u003E\r\n    \u003Cp id=block_progress\u003E \u003C\u002Fp\u003E\r\n    \u003Cp\u003E Please take a short 20 second break.\u003Cbr\u003E\r\n    You may continue once the timer concludes.\r\n    \u003C\u002Fp\u003E\r\n  \u003C\u002Fdiv\u003E\r\n\u003C\u002Fmain\u003E\r\n\r\n\u003Cfooter\u003E\r\n\u003Cdiv class=\"round-time-bar\" data-style=\"smooth\" style=\"--duration: 5;\"\u003E\r\n  \u003Cdiv\u003E\u003C\u002Fdiv\u003E\r\n\u003C\u002Fdiv\u003E\r\n\u003C\u002Ffooter\u003E\r\n\r\n\r\n\u003C!-- bar styles --\u003E\r\n\u003C!-- \u003Cmain\u003E\r\n\u003Cdiv\u003E\r\n  \u003Cdiv class=\"round-time-bar\" style=\"--duration: 10;\"\u003E\r\n    \u003Cdiv\u003E\u003C\u002Fdiv\u003E\r\n  \u003C\u002Fdiv\u003E\r\n\u003C\u002Fdiv\u003E --\u003E\r\n\u003C!-- \u003Cdiv class=\"round-time-bar\" style=\"--duration: 5;\"\u003E\r\n  \u003Cdiv\u003E\u003C\u002Fdiv\u003E\r\n\u003C\u002Fdiv\u003E\r\n\r\n\u003Cdiv class=\"round-time-bar\" data-style=\"smooth\" style=\"--duration: 5;\"\u003E\r\n  \u003Cdiv\u003E\u003C\u002Fdiv\u003E\r\n\u003C\u002Fdiv\u003E\r\n\r\n\u003Cdiv class=\"round-time-bar\" data-color=\"blue\" style=\"--duration: 12;\"\u003E\r\n  \u003Cdiv\u003E\u003C\u002Fdiv\u003E\r\n\u003C\u002Fdiv\u003E\r\n\r\n\u003Cdiv class=\"round-time-bar\" data-color=\"blue\" data-style=\"fixed\" style=\"--duration: 3;\"\u003E\r\n  \u003Cdiv\u003E\u003C\u002Fdiv\u003E\r\n\u003C\u002Fdiv\u003E\r\n\r\n\u003Cbutton id=\"restart-button\"\u003Erestart timers\u003C\u002Fbutton\u003E --\u003E\r\n\u003C\u002Fmain\u003E",
            "timeout": "5000"
          },
          {
            "type": "lab.html.Screen",
            "files": {},
            "responses": {
              "keypress(Space)": "continue"
            },
            "parameters": {},
            "messageHandlers": {
              "run": function anonymous(
) {
if ( ! window.takeBreak) {
  this.end();
} else if (window.endExperiment){
  this.end();
}

}
            },
            "title": "break_continue",
            "content": "\u003Cmain class=\"content-horizontal-center content-vertical-center\"\u003E \r\n  \u003Cdiv\u003E\r\n    \u003Cp id=\"block_progress\"\u003E \u003C\u002Fp\u003E\r\n    \u003Cp\u003EPlease press the \u003Ckbd\u003E\u003Cstrong\u003Espacebar\u003C\u002Fstrong\u003E\u003C\u002Fkbd\u003E to begin the next block.\u003C\u002Fp\u003E\r\n  \u003C\u002Fdiv\u003E\r\n\u003C\u002Fmain\u003E\r\n\r\n\u003Cfooter class=\"content-horizontal-center content-vertical-center\"\u003E\r\n  \u003Cp id=\"foot_instruction\"\u003E \u003C\u002Fp\u003E\r\n\u003C\u002Ffooter\u003E"
          }
        ]
      }
    },
    {
      "type": "lab.html.Screen",
      "files": {},
      "responses": {
        "": ""
      },
      "parameters": {},
      "messageHandlers": {},
      "title": "Exit",
      "content": "\u003Cheader\u003E\r\n  \u003Ch1\u003EThank you!\u003C\u002Fh1\u003E\r\n\u003C\u002Fheader\u003E\r\n\u003Cmain class=\"content-vertical-center content-horizontal-center\"\u003E\r\n  \u003Cdiv align=\"center\"\u003E\r\n    \u003Cp\u003EThe experiment has ended.\u003Cbr\u003E\r\n    Thank you for participating.\r\n    \u003C\u002Fp\u003E\r\n    \u003Cp\u003EYour responses have been recorded. You may now close the browser window to exit.\r\n    \u003C\u002Fp\u003E\r\n  \u003C\u002Fdiv\u003E\r\n\u003C\u002Fmain\u003E\r\n"
    }
  ]
})

// Let's go!
study.run()