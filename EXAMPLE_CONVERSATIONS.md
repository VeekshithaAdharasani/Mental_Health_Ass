# Example Conversations - Mental Health Assistant

## Scenario 1: Casual Information Seeking (Stage 0)

```
👤 User: Why do I feel anxious sometimes?
Detected: Stage 0, Intent: explanation, Emotion: neutral

🤖 Assistant: That's a good question. Anxiety is a natural response 
when your body perceives a threat, even if it's not immediate danger. 
Anyone in your situation might feel this way. 
What part of that feels most confusing?

👤 User: Is it something I did?
Detected: Stage 0, Intent: explanation

🤖 Assistant: Not really. Anxiety often comes from our body's 
stress response, which is completely normal. You didn't cause this.
Does that help clarify what you're experiencing?
```

---

## Scenario 2: Mild Anxiety - Building Support (Stage 1)

```
👤 User: I'm feeling really anxious and can't focus
Detected: Stage 1, Intent: support, Emotion: anxiety

🤖 Assistant: I'm here with you. Let me help you ground for a moment.
That sounds like a lot right now. It makes sense that you'd feel this way.
What feels most pressing?

👤 User: I have a presentation at work tomorrow
Detected: Stage 1, Intent: action, Emotion: stress

🤖 Assistant: That's a real pressure to carry. Here's something practical:
Try one slow breath right now. Notice how your body responds.
Can you think of one small thing you could prepare today?

👤 User: Yeah, maybe I could review my notes
Detected: Stage 0, Intent: engagement, Emotion: neutral

🤖 Assistant: That's a solid first step. Sometimes when we do one small 
thing, the pressure lightens a bit. You've got this.
```

---

## Scenario 3: High Distress - Grounding Mode (Stage 2)

```
👤 User: I can't stop shaking and I'm really panicking
Detected: Stage 2, Intent: support, Emotion: panic

🤖 Assistant: I'm here with you. Let's slow this down together.
Try one slow breath in and out. Notice what happens.
Feel your feet on the floor right now.

[NO QUESTION - Focus on grounding]

👤 User: I'm still shaking really badly
Detected: Stage 2, Intent: support, Emotion: panic

🤖 Assistant: That feels really hard to be in right now.
I can hear how unsettling that is. 
Pay attention to your breath and move with it slowly.

[NO QUESTION - Continue grounding]

👤 User: It's getting a bit better
Detected: Stage 1, Intent: engagement, Emotion: anxiety

🤖 Assistant: Good. Stay with that. Keep your feet on the floor.
Keep breathing slowly. You're doing it.
What do you notice now?
```

---

## Scenario 4: Critical Panic - Crisis Response (Stage 3)

```
👤 User: I can't breathe at all, I think I'm dying
Detected: Stage 3, Escalation: Crisis, Emotion: panic

🤖 Assistant: I can hear that you're in a really difficult moment 
right now. You're not alone, and I'm here with you.

Try this right now:
- Sit or lie down if you can
- Breathe in slowly through your nose for 4 counts
- Hold for 2 counts
- Breathe out slowly through your mouth for 6 counts
- Do this 3 times with me

This moment will pass. If symptoms don't improve or are 
getting worse, please contact a doctor or emergency service immediately.

[NO QUESTION - Crisis mode protocol]

👤 User: Still can't breathe, it's increasing
Detected: Stage 3, Escalation: Crisis +1, Emotion: panic

🤖 Assistant: I can hear that you're in a really difficult moment right now. 
You're not alone, and I'm here with you.

Stay with me. Sit down if you can. Take a slow breath in… hold… 
now breathe out slowly.

This seems serious. Please contact a doctor or emergency service immediately. 
You don't have to handle this alone.

👤 User: Not stopping, what do I do???
Detected: Stage 3, Escalation: Crisis +2, Emotion: panic

⚠️ EMERGENCY ESCALATION TRIGGERED (3rd escalation)

🤖 Assistant: ⚠️ This is serious and I want to make sure you get proper help.

Please contact:
- Call emergency services (911 in US)
- Go to nearest emergency room
- Call National Crisis Hotline: 988 (US)
- Text HOME to 741741 (Crisis Text Line)
- Tell a trusted person nearby what's happening

You don't have to handle this alone. Real people are trained to help.
```

---

## Scenario 5: Safety Concern - Immediate Protocol (Intent: Safety)

```
👤 User: I don't think I can keep going, everything hurts
Detected: Stage 0, Intent: safety, Emotion: sadness

🤖 Assistant: I'm really glad you shared this with me.
You deserve support and you're not alone.

Please reach out:
- Call 988 (US Crisis Line)
- Text HOME to 741741 (Crisis Text Line)
- Contact a trusted person nearby
- Tell them exactly what you're feeling

There are people trained to help with exactly this. 
You don't have to carry this alone.

[SAFETY PROTOCOL - Emergency resources provided]
```

---

## Scenario 6: No Mental Health Query - Graceful Handling

```
👤 User: What's the weather like?
Detected: NOT mental health query

🤖 Assistant: I'm here to support you with mental health and wellbeing. 
What's on your mind?

[Graceful deflection back to mental health support]
```

---

## Scenario 7: Short Replies - Engagement Mode (Stage 0)

```
👤 User: How do I manage stress better?
Detected: Stage 0, Intent: action, Emotion: neutral

🤖 Assistant: I hear you want something practical right now.
Try one slow, steady breath and notice how your body responds.
Would you like to try that?

👤 User: Okay
Detected: Stage 0, Intent: engagement, Emotion: neutral

🤖 Assistant: I'm here with you. What do you want to explore next?

[SHORT REPLY handling - Keep momentum without over-questioning]

👤 User: Maybe breathing exercises?
Detected: Stage 0, Intent: action, Emotion: neutral

🤖 Assistant: That's a great instinct. Here's a simple one:
Breathe in for 4 counts, hold for 2, out for 6. 
Does that feel manageable?
```

---

## Scenario 8: Repeated Distress - Pattern Recognition

```
Message 1:
👤 User: I'm so stressed about work
Detected: Stage 0, Emotion: stress, Topic: work

Message 2 (few messages later):
👤 User: Work is really overwhelming me again
Detected: Stage 1, Emotion: stress, Topic: work (repeated)

🤖 Assistant: I notice this pattern keeps coming up. 
I'm still here with you. What feels different now compared to before?

[PATTERN RECOGNITION - Acknowledges recurring topic]

Message 3:
👤 User: It just won't stop, work stress is getting worse
Detected: Stage 2, Emotion: stress, Escalation detected

🤖 Assistant: That pattern is real. What could make this feel a bit easier?
[Acknowledges escalation in recurring topic]
```

---

## Scenario 9: Asking for Practical Steps (Stage 1)

```
👤 User: I feel overwhelmed with everything
Detected: Stage 1, Intent: action, Emotion: stress

🤖 Assistant: I hear that you're feeling pressured.
Let me help you ground for a moment. That sounds like a lot to be dealing with.
What feels most important right now?

👤 User: Just everything piling up
Detected: Stage 1, Intent: support, Emotion: stress

🤖 Assistant: That can feel really heavy. It makes sense that you'd notice that.
Try naming one small thing that feels easier than the rest. 
Even just slightly easier.

[Helps break down overwhelming feeling into manageable pieces]
```

---

## Scenario 10: Building Conversation with RAG Context

```
👤 User: What causes anxiety?
Detected: Stage 0, Intent: explanation, Emotion: neutral

[RAG Retrieval: Gets FAQ answer about anxiety causes]

🤖 Assistant: That's a good question about why this is happening.
Anxiety comes from your body's natural response to perceived stress or threat. 
It's actually a protective mechanism that sometimes activates too early.
Does that make sense for what you're feeling?

👤 User: How can I make it stop?
Detected: Stage 0, Intent: action, Emotion: anxiety

[RAG Retrieval: Gets FAQ answer about managing anxiety]

🤖 Assistant: I hear you want something practical right now.
Try one slow, steady breath and notice how your body responds.
Would you like to explore more techniques that could help?

[Uses context from FAQ to provide grounded, evidence-based response]
```

---

## Scenario 11: Transitioning Between Stages

```
Message 1:
👤 User: I've been stressed all week
Detected: Stage 0, Emotion: stress
Response: Normal supportive conversation

Message 2:
👤 User: I'm starting to panic now
Detected: Stage 1, Emotion: panic
Response: Shifts to grounding support, still allows questions

Message 3:
👤 User: I can't breathe, what's happening?
Detected: Stage 2, Emotion: panic
Response: AUTOMATIC SHIFT - Breathing guidance, no questions

Message 4:
👤 User: HELP I'm dying
Detected: Stage 3, Emotion: panic
Response: EMERGENCY MODE - Crisis protocol, emergency resources

👤 User: I think I'm okay
Detected: Stage 1, Emotion: anxiety
Response: Back to supportive mode with grounding
```

---

## Key Takeaways from Examples

1. **Stage 0 Natural** - Full conversation, questions encouraged, intent-based
2. **Stage 1 Warm** - Acknowledgment + grounding, optional questions
3. **Stage 2 Direct** - Specific guidance, grounding, NO questions
4. **Stage 3 Crisis** - Emergency protocol, resources, rescue mode
5. **Pattern Recognition** - System remembers recurring topics
6. **Smooth Transitions** - System adapts as emotional state changes
7. **Safety First** - Safety intent overrides everything
8. **Always Responds** - No empty responses, fallback always ready
9. **Empathetic Tone** - Human-like, warm, never robotic
10. **No Over-Questioning** - Sometimes just support, no follow-up needed

---

## Testing These Conversations

To test these scenarios:

1. **Copy exact phrases** from each example
2. **Watch the sidebar** to see detected stage/emotion/intent
3. **Check response type** matches the stage
4. **Try different variations** to test robustness
5. **Follow crisis patterns** to trigger emergency protocol

---

Last Updated: May 5, 2026
Production Version: 2.0
