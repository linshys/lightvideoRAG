import asyncio
import traceback

class InsertTaskManager:
    def __init__(self, loop=None):
        self.loop = loop or self._get_loop()
        self.tasks = []

    def _get_loop(self):
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def submit(self, coro):
        """提交一个协程任务，立即调度但不等待"""
        task = self.loop.create_task(coro)
        self.tasks.append(task)

    async def wait_all(self):
        """等待所有任务完成，并打印异常信息"""
        if not self.tasks:
            return

        print(f"🟢 等待 {len(self.tasks)} 个插入任务完成...")
        done, pending = await asyncio.wait(self.tasks)

        failed = 0
        for task in done:
            if task.exception():
                failed += 1
                print("❌ 插入任务失败:", task.exception())
                traceback.print_exception(type(task.exception()), task.exception(), task.exception().__traceback__)

        print(f"✅ 插入任务完成：成功 {len(done)-failed} / 失败 {failed}")
        self.tasks.clear()
