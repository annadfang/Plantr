class EventManager {

  constructor() {      
    this.eventLookup = {};
  }

  off(event, callback) {
    var listeners = this.eventLookup[event];                
    if (event === "*") this.eventLookup = {}; 
    else if(!callback) this.eventLookup[event] = []; 
    else _.remove(listeners, { callback });
  }

  on(event, callback, scope) {
    var listeners = this.eventLookup[event];
    if (!listeners) this.eventLookup[event] = listeners = [];  
    listeners.push({ callback, scope });      
    return () => _.remove(listeners, { callback });
  }

  fire(event, ...data) {
    var listeners = this.eventLookup[event];
    if (!listeners) return;
    listeners.forEach(list => {
      try {
        return list.callback.apply(list.scope, data);
      } catch(e) {
        return _.isError(e) ? e : new Error(e);
      }
    });      
  }
}

var events = new EventManager();

var ns = "http://www.w3.org/2000/svg";
var d  = "M0,0 Q5,-5 10,0 5,5 0,0z";

var stems  = $("#stems");
var leaves = $("#leaves");
var svg    = $("svg");

var leafCount = 15;
var plants    = 5;
var centerX   = 200;
var offsetX   = 175;

generate();

function generate() {
      
  _.times(plants, createPlant);
    
  stems.children().each(function() {
    
    var tween = TweenMax.to(this, _.random(2, 4), {
      drawSVG: true,    
      delay: _.random(2),
      onStart: () => TweenLite.set(this, { opacity: 1 }),
      onUpdate: () => events.fire(this.id, tween.progress())      
    });
  });  
}

function createPlant() {
  var points;
  var x = _.random(centerX - offsetX, centerX + offsetX);
  var y = 400;
  var count  = _.random(30, 45);
  var points = [{ x, y }];
    
  for (var i = 1; i <= count; i++) {
    points.push({
      x: points[i - 1].x + i * .005 * (_.random(15) - 10),
      y: 300 - 5 * i
    });
  }
  var stem   = $(document.createElementNS(ns, "path")).appendTo(stems);
  var length = points.length;  
  var values = points.map(point => `${point.x},${point.y}`);
  var height = points[length - 1].y;   
  var id     = _.uniqueId("grow"); 
  
  TweenLite.set(stem, {     
    opacity: 0,
    drawSVG: 0,
    attr: { id, d: `M${values.join(" ")}` }
  });
  
  for (var i = 0; i < leafCount; i++) {
    var point = points[length - 1 - i];    
    var scale = {
      x: 1 + 0.1 * i,
      y: 1 + 0.05 * i
    };
    createLeaf(point, scale, height, id);
  }    
}

function createLeaf(point, scale, height, grow) {
  
  var leaf  = $(document.createElementNS(ns, "path")).appendTo(leaves);  
  var start = height / point.y;  
  var off   = events.on(grow, growLeaf);
  
  function growLeaf(growth) {
      off();
      TweenLite.set(leaf, {x: point.x, y: point.y, scaleX: scale.x,
        scaleY: scale.y, rotation: _.random(180) - 180, fill: `rgb(0,${_.random(80, 160)},0)`,
        attr: { d }        
      });
      
      TweenLite.from(leaf, 1, { scale: 0 });
  }              
}

