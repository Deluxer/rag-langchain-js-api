import { Controller, Get } from '@nestjs/common';
import { AppService } from './app.service';

@Controller('llm')
export class AppController {
  constructor(private readonly appService: AppService) {}

  @Get('gpt3')
  gp3() {
    return this.appService.gpt3();
  }

  @Get('mistral')
  mistral() {
    return this.appService.mistral();
  }
}
