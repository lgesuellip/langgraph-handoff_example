from __future__ import annotations
from src.agents.graph import graph
from langsmith import Client, traceable
from langsmith.evaluation import aevaluate
from langsmith.schemas import Example, Run
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage
import asyncio

class TrajectoryEvaluator:
    def __init__(self):
        self.client = Client()

    @traceable
    def build_trajectory_and_tool_calls(self, message_dict: dict) -> Dict[str, Any]:
        """Build combined dataset with trajectory and tool calls"""
        def extract_trajectory_and_tools(messages: List[dict]) -> tuple:
            trajectory = []
            tool_calls = []
            
            for msg in messages:
                msg_type = msg.get("type") if isinstance(msg, dict) else msg.type
                msg_tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else msg.additional_kwargs.get("tool_calls")
                
                if msg_type == "ai":
                    if msg_tool_calls:
                        for tool_call in msg_tool_calls:
                            # Handle both direct tool calls and function-wrapped tool calls
                            if "function" in tool_call:
                                name = tool_call["function"]["name"]
                                args = tool_call["function"]["arguments"]
                                if isinstance(args, str):
                                    import json
                                    args = json.loads(args)
                            else:
                                name = tool_call["name"]
                                args = tool_call["args"]
                                
                            trajectory.append(name)
                            tool_calls.append({
                                "name": name,
                                "args": args
                            })
            
            return trajectory, tool_calls

        # Handle both dict and list of BaseMessage objects
        messages = message_dict.get("messages", []) if isinstance(message_dict, dict) else message_dict
        trajectory, tool_calls = extract_trajectory_and_tools(messages)
        
        return {
            "trajectory": trajectory,
            "tool_calls": tool_calls,
        }

    @traceable
    def evaluate_tool_args(self, reference_tools: List[dict], output_tools: List[dict]) -> float:
        """Evaluates if reference tool arguments appear in the output tools"""
        try:
            for ref_tool in reference_tools:
                found_match = False
                for out_tool in output_tools:
                    if ref_tool["name"] == out_tool["name"] and ref_tool["args"] == out_tool["args"]:
                        found_match = True
                        break
                if not found_match:
                    return 0.0
            return 1.0
        except Exception:
            return 0.0

    @traceable
    def evaluate_tool_order(self, reference_tools: List[dict], output_tools: List[dict]) -> float:
        """Evaluates if tools are called in the exact order as reference"""
        reference_names = [tool["name"] for tool in reference_tools]
        output_names = [tool["name"] for tool in output_tools]
        
        # Check if all reference tools appear in output in the same order
        ref_iter = iter(reference_names)
        try:
            return 1.0 if all(name in output_names for name in ref_iter) else 0.0
        except StopIteration:
            return 0.0

    @traceable
    def evaluate_agent(self, run: Run, example: Example) -> Dict[str, Any]:
        """Evaluates trajectory, tool arguments, and tool order"""
        reference = self.build_trajectory_and_tool_calls(example.outputs.get("result"))
        output = self.build_trajectory_and_tool_calls(run.outputs.get("result", {}))
        
        args_score = self.evaluate_tool_args(reference["tool_calls"], output["tool_calls"])
        order_score = self.evaluate_tool_order(reference["tool_calls"], output["tool_calls"])
        
        return [
            {"score": args_score, "key": "tool_args"},
            {"score": order_score, "key": "tool_order"}
        ]

    @staticmethod
    async def run_graph(example: dict):
        """Run the graph for a given example"""
        msg = {"messages": HumanMessage(content=example["query"])}
        messages = await graph.ainvoke(msg)
        return {"result": messages}

    async def run_evaluation(self, version: str = "0.1.0", max_concurrency: int = 5):
        """Run the full evaluation pipeline"""
        _ = await aevaluate(
            self.run_graph,
            data=self.client.list_examples(dataset_name="evals_example", splits=["trajectory"]),
            evaluators=[self.evaluate_agent],
            experiment_prefix="researcher:trajectory",
            max_concurrency=max_concurrency,
            metadata={
                "version": version,
            },
        )

async def main():
    evaluator = TrajectoryEvaluator()
    await evaluator.run_evaluation()

if __name__ == "__main__":
    asyncio.run(main())